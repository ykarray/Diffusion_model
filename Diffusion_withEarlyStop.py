#%% --------------- IMPORTS & CONFIGURATION ---------------
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import sys
from datetime import datetime
import seaborn as sns
import lpips

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path_data = "C:/Users/ykarr/Desktop/model_Diffusion_gen/Data"

def visualize_forward_process(config, diffusion, epoch):
    with rasterio.open(path_data + "/train/sequence_0/sequence_0.tif") as src:
        img = torch.tensor(src.read()).float().to(device)

    timesteps = [0, 500, 1000, 1500, 1999]
    plt.figure(figsize=(15, 3))

    for i, t in enumerate(timesteps):
        noised_img, _, _ = diffusion.forward_diffusion(  # Updated to handle new return values
            img.unsqueeze(0),
            torch.tensor([t], device=device))
        plt.subplot(1, len(timesteps), i+1)
        plt.imshow(noised_img[0,0].cpu().numpy(), cmap='gray')
        plt.title(f"t={t}")
        plt.axis('off')

    plt.savefig(config.output_dir / f"forward_process_epoch_{epoch}.png")
    plt.close()

#%% --------------- DATA STRUCTURE ANALYSIS ---------------
def analyze_data_structure(data_root):
    try:
        train_dir = Path(data_root) / 'train'
        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        sequences = [d for d in train_dir.iterdir() if d.is_dir()]
        if not sequences:
            raise ValueError("No sequences found in training directory")

        sample_seq = sequences[0]
        tif_path = sample_seq / f"{sample_seq.name}.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"Sample TIFF not found: {tif_path}")

        with rasterio.open(tif_path) as src:
            num_bands, height, width = src.count, src.height, src.width

        context_path = sample_seq / "context.pt"
        if not context_path.exists():
            raise FileNotFoundError(f"Context file not found: {context_path}")

        context_sample = torch.load(context_path, map_location="cpu")

        return {
            "num_sequences": len(sequences),
            "num_bands": num_bands,
            "img_dims": (height, width),
            "context_dims": context_sample.shape
        }
    except Exception as e:
        print(f"Data structure analysis failed: {str(e)}")
        raise

#%% --------------- DYNAMIC CONFIGURATION ---------------
class DynamicConfig:
    def __init__(self, data_specs):
        self.data_root = Path(path_data)
        self.output_dir = self.data_root / "DiffusionEarly"
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.generated_dir = self.output_dir / "generated"
        self.generated_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        self.num_sequences = data_specs["num_sequences"]
        self.num_bands = data_specs["num_bands"]
        self.img_dims = data_specs["img_dims"]
        self.context_dims = data_specs["context_dims"]
        self.T = 2000
        self.base_channels = 256
        self.learning_rate = 2e-4
        self.beta_schedule_type = "cosine"
        self.early_stopping_patience = 10  # Number of epochs to wait before stopping
        self.early_stopping_min_delta = 0.001 
        self.batch_size = self._calculate_batch_size()
        self.epochs = self._calculate_epochs()
        self.num_downsample = self._calculate_downsample_layers()

        self.beta = self._create_beta_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.image_mean = None
        self.image_std = None
        self.context_mean = None
        self.context_std = None
        print(f"Final alpha_bar value: {self.alpha_bar[-1].item():.5f}")
        if self.alpha_bar[-1] > 0.01:
            warnings.warn("Insufficient total noise - adjust beta schedule parameters!")
        print(f"Alpha bar at t=500: {self.alpha_bar[500].item():.5f}")
        print(f"Alpha bar at t=1000: {self.alpha_bar[1000].item():.5f}")
        print(f"Alpha bar at t=1500: {self.alpha_bar[1500].item():.5f}")
        print(f"Alpha bar at t=1999: {self.alpha_bar[1999].item():.5f}")

    def _calculate_batch_size(self):
        if self.num_sequences > 1000:
            return 32
        elif self.num_sequences > 500:
            return 16
        return 8

    def _calculate_epochs(self):
        base_epochs = max(500, int(100000 / self.num_sequences))
        return min(base_epochs, 1000)

    def _calculate_downsample_layers(self):
        min_dim = min(self.img_dims)
        base_layers = int(math.log2(min_dim // 16))
        if self.num_sequences < 500:
            return max(3, base_layers - 1)
        elif self.num_sequences < 1000:
            return base_layers
        else:
            return min(base_layers + 1, 5)

    def _create_beta_schedule(self):
        if self.beta_schedule_type == "cosine":
            return self._cosine_beta_schedule()
        return self._linear_beta_schedule()

    def _linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.T)

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.T + 1
        x = torch.linspace(0, self.T, steps)
        alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.5)

#%% --------------- ADAPTIVE DATASET ---------------
class DiffusionDataset(Dataset):
    def __init__(self, config, normalize_before_diff=False):
        self.config = config
        self.normalize_before_diff = normalize_before_diff
        self.train_dir = config.data_root / "train"
        self.sequences = self._get_valid_sequences()
        self._compute_statistics()

        self.pad_factor = 2 ** self.config.num_downsample
        self.pad_h = (self.pad_factor - (self.config.img_dims[0] % self.pad_factor)) % self.pad_factor
        self.pad_w = (self.pad_factor - (self.config.img_dims[1] % self.pad_factor)) % self.pad_factor

        self.config.pad_h = self.pad_h
        self.config.pad_w = self.pad_w
        self.config.padded_dims = (
            self.config.img_dims[0] + self.pad_h,
            self.config.img_dims[1] + self.pad_w
        )

        self.config.image_mean = self.image_mean
        self.config.image_std = self.image_std
        self.config.context_mean = self.context_mean
        self.config.context_std = self.context_std

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        try:
            with rasterio.open(self.train_dir / seq / f"{seq}.tif") as src:
                img_data = src.read()
                if len(img_data.shape) == 2:
                    img_data = np.expand_dims(img_data, axis=0)
                img = torch.tensor(img_data.astype(np.float32))

            if self.normalize_before_diff:
                img = (img - self.config.image_mean.cpu().view(-1, 1, 1)) / self.config.image_std.cpu().view(-1, 1, 1)

            img_diff = img

            img_diff = F.pad(img_diff, (0, self.pad_w, 0, self.pad_h))
            img_diff = (img_diff - self.config.image_mean.cpu().view(-1, 1, 1)) / self.config.image_std.cpu().view(-1, 1, 1)

            context = torch.load(self.train_dir / seq / "context.pt", map_location="cpu")

            if len(context.shape) == 2:
                context = context.unsqueeze(0)

            context_diff = context

            if context_diff.shape[-2:] == img_diff.shape[-2:]:
                context_diff = F.pad(context_diff, (0, self.pad_w, 0, self.pad_h))
                context_diff = (context_diff - self.config.context_mean.cpu().view(-1, 1, 1)) / self.config.context_std.cpu().view(-1, 1, 1)

            return img_diff, context_diff

        except Exception as e:
            print(f"Error loading sequence {seq}: {str(e)}")
            dummy_img = torch.zeros((self.config.num_bands, *self.config.padded_dims))
            dummy_ctx = torch.zeros((self.config.context_dims[0], *self.config.padded_dims))
            return dummy_img, dummy_ctx

    def _get_valid_sequences(self):
        valid_seqs = []
        for seq_dir in self.train_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            try:
                tif_path = seq_dir / f"{seq_dir.name}.tif"
                context_path = seq_dir / "context.pt"
                if tif_path.exists() and context_path.exists():
                    valid_seqs.append(seq_dir.name)
            except Exception as e:
                print(f"Skipping invalid sequence {seq_dir.name}: {str(e)}")
        return valid_seqs

    def _compute_statistics(self):
        image_sum = torch.zeros(self.config.num_bands)
        image_sq_sum = torch.zeros(self.config.num_bands)
        context_sum = torch.zeros(self.config.context_dims[0])
        context_sq_sum = torch.zeros(self.config.context_dims[0])
        valid_count = 0

        for seq in tqdm(self.sequences, desc="Computing statistics"):
            try:
                with rasterio.open(self.train_dir / seq / f"{seq}.tif") as src:
                    img_data = src.read()
                    if len(img_data.shape) == 2:
                        img_data = np.expand_dims(img_data, axis=0)
                    img = torch.tensor(img_data.astype(np.float32))

                img_diff = img
                image_sum += img_diff.mean(dim=(1, 2))
                image_sq_sum += (img_diff ** 2).mean(dim=(1, 2))

                context = torch.load(self.train_dir / seq / "context.pt", map_location="cpu")

                if len(context.shape) == 2:
                    context = context.unsqueeze(0)

                context_diff = context
                context_sum += context_diff.mean(dim=(1, 2))
                context_sq_sum += (context_diff ** 2).mean(dim=(1, 2))

                valid_count += 1
            except Exception as e:
                print(f"Skipping corrupted sequence {seq}: {str(e)}")

        if valid_count == 0:
            raise ValueError("No valid sequences found for statistics calculation")

        self.image_mean = (image_sum / valid_count).to(device)
        self.image_std = ((image_sq_sum/valid_count - self.image_mean.cpu()**2).sqrt()).to(device)
        self.context_mean = (context_sum / valid_count).to(device)
        self.context_std = ((context_sq_sum/valid_count - self.context_mean.cpu()**2).sqrt()).to(device)

    def __len__(self):
        return len(self.sequences)

#%% --------------- DYNAMIC ARCHITECTURE ---------------
def create_diffusion_model(config):
    class TimeEmbedding(nn.Module):
        def __init__(self, time_dim, out_dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(time_dim, out_dim * 4),
                nn.GELU(),
                nn.Linear(out_dim * 4, out_dim))

        def forward(self, x):
            return self.mlp(x)

    class DynamicBlock(nn.Module):
        def __init__(self, in_ch, out_ch, time_dim):
            super().__init__()
            self.time_emb = TimeEmbedding(time_dim, out_ch)
            self.group_norm1 = nn.GroupNorm(min(8, out_ch//4), out_ch)
            self.group_norm2 = nn.GroupNorm(min(8, out_ch//4), out_ch)

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                self.group_norm1,
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                self.group_norm2,
                nn.GELU()
            )
            self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x, t_emb):
            t_emb = self.time_emb(t_emb)
            h = self.conv(x) + self.residual(x)
            return h + t_emb.unsqueeze(-1).unsqueeze(-1)

    class AttentionBlock(nn.Module):
        def __init__(self, channels, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.norm = nn.GroupNorm(min(8, channels//4), channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.proj = nn.Conv2d(channels, channels, 1)

        def forward(self, x):
            B, C, H, W = x.shape
            qkv = self.qkv(self.norm(x))
            q, k, v = qkv.reshape(B, self.num_heads, C // self.num_heads * 3, H * W).chunk(3, dim=2)
            scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
            attn = torch.einsum("bhdn,bhdm->bhmn", q * scale, k * scale)
            attn = attn.softmax(dim=-1)
            h = torch.einsum("bhmn,bhdm->bhdn", attn, v)
            h = h.reshape(B, C, H, W)
            return self.proj(h) + x

    class DynamicUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.time_dim = 128
            self.base_channels = config.base_channels
            self.num_downsample = config.num_downsample

            self.time_mlp = nn.Sequential(
                nn.Linear(self.time_dim, self.time_dim * 4),
                nn.GELU(),
                nn.Linear(self.time_dim * 4, self.time_dim)
            )

            self.init_conv = nn.Conv2d(config.num_bands, self.base_channels, 3, padding=1)
            self.down_blocks = nn.ModuleList()
            current_ch = self.base_channels

            for _ in range(self.num_downsample):
                self.down_blocks.append(nn.Sequential(
                    DynamicBlock(current_ch, current_ch * 2, self.time_dim),
                    nn.MaxPool2d(2)
                ))
                current_ch *= 2

            self.mid_block = nn.Sequential(
                DynamicBlock(current_ch, current_ch, self.time_dim),
                AttentionBlock(current_ch)
            )

            self.context_encoder = nn.Sequential(
                nn.Conv2d(config.context_dims[0], 64, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((
                    config.padded_dims[0] // (2 ** config.num_downsample),
                    config.padded_dims[1] // (2 ** config.num_downsample)
                )),
                nn.Conv2d(64, current_ch, 1)
            )

            self.up_blocks = nn.ModuleList()
            for i in range(self.num_downsample):
                skip_ch = self.base_channels * (2 ** (self.num_downsample - i))
                in_ch = (current_ch // 2) + skip_ch
                self.up_blocks.append(nn.Sequential(
                   nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),nn.Conv2d(current_ch, current_ch // 2, 3, padding=1)) ,
                    DynamicBlock(in_ch, current_ch // 2, self.time_dim)
                ))
                current_ch = current_ch // 2

            # Modified to output both noise and x0 predictions
            self.final_conv = nn.Conv2d(current_ch, config.num_bands * 2, 3, padding=1)

        def forward(self, x, t, context):
            t_emb = self._timestep_embedding(t)
            t_emb = self.time_mlp(t_emb)

            context_feat = self.context_encoder(context)

            x = self.init_conv(x)
            skips = []

            for block in self.down_blocks:
                x = block[0](x, t_emb)
                skips.append(x)
                x = block[1](x)

            x = self.mid_block[0](x, t_emb)
            x = self.mid_block[1](x) + context_feat

            for block in self.up_blocks:
                x = block[0](x)
                x = torch.cat([x, skips.pop()], dim=1)
                x = block[1](x, t_emb)

            output = self.final_conv(x)
            noise_pred, x0_pred = torch.chunk(output, 2, dim=1)  # Split into noise and x0 predictions
            return noise_pred, x0_pred

        def _timestep_embedding(self, t):
            half_dim = self.time_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = t.float()[:, None] * emb[None, :]
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
            return emb

    return DynamicUNet()

#%% --------------- DIFFUSION PROCESS (IMPROVED) ---------------
class DiffusionProcess(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eta = 0.5
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.config.alpha))
        self.register_buffer('sqrt_recipm1_alphas', torch.sqrt(1.0 / self.config.alpha - 1))

    def forward_diffusion(self, x0, t):
        x0 = x0.float()
        sqrt_alpha_bar = torch.sqrt(self.config.alpha_bar[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.config.alpha_bar[t])[:, None, None, None]
        noise = torch.randn_like(x0)
        noise = noise * sqrt_one_minus_alpha_bar.mean()  # Scale noise
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise, x0  # Return original x0 for loss calculation

    def reverse_diffusion_step(self, model, xt, t, context, use_ddim=True):
        batch_size = xt.shape[0]
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            pred_noise, pred_x0_direct = model(xt, t_tensor, context)
            pred_x0_from_noise = self.predict_x0_from_noise(xt, t_tensor, pred_noise)
            pred_x0 = 0.5 * pred_x0_direct + 0.5 * pred_x0_from_noise  # Combine predictions

        if t == 0 or use_ddim:
            return pred_x0
        else:
            alpha_bar_t = self.config.alpha_bar[t_tensor]
            alpha_bar_prev = self.config.alpha_bar[t_tensor-1]
            sigma_t = self.eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))

            eps = torch.randn_like(xt)
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * pred_noise
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * eps
            return x_prev

    def predict_x0_from_noise(self, xt, t, noise):
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        sqrt_recipm1_alpha = self.sqrt_recipm1_alphas[t]
        return sqrt_recip_alpha[:, None, None, None] * xt - sqrt_recipm1_alpha[:, None, None, None] * noise

#%% --------------- TRAINING LOOP ---------------
def plot_loss_curves(log_path, output_dir):
    data = np.genfromtxt(log_path, delimiter=',', skip_header=1)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    epochs = data[:, 0]
    train_loss = data[:, 1]
    val_loss = data[:, 2]

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    plt.subplot(121)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, np.log10(train_loss), label='Train Loss')
    plt.plot(epochs, np.log10(val_loss), label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log10(Loss)')
    plt.title('Log-Scaled Loss')
    plt.legend()

    plt.tight_layout()
    plot_path = output_dir / "training_plots.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training plots to {plot_path}")

def train_model(config):
    full_dataset = DiffusionDataset(config)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        [x for x in train_dataset if x is not None],
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2
    )
    val_loader = DataLoader(
        [x for x in val_dataset if x is not None],
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=2
    )

    model = create_diffusion_model(config).to(device)
    print("Model architecture:")
    print(model)
    with open(config.output_dir / "model_architecture.txt", "w") as f:
        print(model, file=f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    diffusion = DiffusionProcess(config)
    best_val_loss = float('inf')
    log_file = config.logs_dir / "training_log.txt"
    with open(log_file, "w") as f:
        f.write("Epoch,Train Loss,Val Loss,Train PSNR,Val PSNR\n")

    scaler = torch.cuda.amp.GradScaler()
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    try:
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            train_psnr = 0.0
            for x0, context in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
                x0 = x0.to(device)
                context = context.to(device)

                t = torch.randint(0, config.T, (x0.size(0),), device=device)
                xt, noise, clean_x0 = diffusion.forward_diffusion(x0, t)  # Updated to get clean_x0

                # Noise conditioning augmentation
                if torch.rand(1).item() < 0.1:  # 10% chance
                    xt = torch.randn_like(x0)
                    t = torch.full((x0.size(0),), config.T-1, device=device)

                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    pred_noise, pred_x0_direct = model(xt, t, context)
                    
                    # Noise prediction loss
                    noise_loss = F.mse_loss(pred_noise, noise)
                    
                    # Direct x0 prediction loss
                    x0_loss = F.mse_loss(pred_x0_direct, clean_x0)
                    
                    # LPIPS loss for noise prediction
                    lpips_losses = []
                    for band_idx in range(pred_noise.shape[1]):
                        pred_band = pred_noise[:, band_idx:band_idx+1]
                        noise_band = noise[:, band_idx:band_idx+1]
                        pred_band_norm = 2 * (pred_band - pred_band.min()) / (pred_band.max() - pred_band.min() + 1e-8) - 1
                        noise_band_norm = 2 * (noise_band - noise_band.min()) / (noise_band.max() - noise_band.min() + 1e-8) - 1
                        pred_band_norm = pred_band_norm.repeat(1, 3, 1, 1)
                        noise_band_norm = noise_band_norm.repeat(1, 3, 1, 1)
                        lpips_loss = lpips_loss_fn(pred_band_norm, noise_band_norm).mean()
                        lpips_losses.append(lpips_loss)
                    lpips_loss = torch.stack(lpips_losses).mean()
                    
                    # LPIPS loss for x0 prediction
                    x0_lpips_losses = []
                    for band_idx in range(pred_x0_direct.shape[1]):
                        pred_band = pred_x0_direct[:, band_idx:band_idx+1]
                        x0_band = clean_x0[:, band_idx:band_idx+1]
                        pred_band_norm = 2 * (pred_band - pred_band.min()) / (pred_band.max() - pred_band.min() + 1e-8) - 1
                        x0_band_norm = 2 * (x0_band - x0_band.min()) / (x0_band.max() - x0_band.min() + 1e-8) - 1
                        pred_band_norm = pred_band_norm.repeat(1, 3, 1, 1)
                        x0_band_norm = x0_band_norm.repeat(1, 3, 1, 1)
                        lpips_loss = lpips_loss_fn(pred_band_norm, x0_band_norm).mean()
                        x0_lpips_losses.append(lpips_loss)
                    x0_lpips_loss = torch.stack(x0_lpips_losses).mean()
                    
                    # Consistency loss
                    pred_x0_from_noise = diffusion.predict_x0_from_noise(xt, t, pred_noise)
                    consistency_loss = F.mse_loss(pred_x0_direct, pred_x0_from_noise)
                    
                    # Combined loss
                    noise_weight = 1.0
                    x0_weight = 0.5
                    consistency_weight = 0.1
                    loss = (noise_weight * noise_loss) + (x0_weight * x0_loss) + lpips_loss + \
                           (x0_weight * x0_lpips_loss) + (consistency_weight * consistency_loss)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                mse = F.mse_loss(pred_noise, noise)
                psnr = 10 * torch.log10(1 / mse)
                train_psnr += psnr.item()
                train_loss += mse.item()

            model.eval()
            val_loss = 0.0
            val_psnr = 0.0
            with torch.no_grad():
                for x0, context in val_loader:
                    x0 = x0.to(device)
                    context = context.to(device)
                    t = torch.randint(0, config.T, (x0.size(0),), device=device)
                    xt, noise, _ = diffusion.forward_diffusion(x0, t)
                    pred_noise, _ = model(xt, t, context)  # Only need noise for validation
                    mse = F.mse_loss(pred_noise, noise)
                    psnr = 10 * torch.log10(1 / mse)
                    val_psnr += psnr.item()
                    val_loss += mse.item()
                    print("Input range:", x0.min().item(), x0.max().item())
                    print("Output range:", pred_noise.min().item(), pred_noise.max().item())

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_psnr = train_psnr / len(train_loader)
            avg_val_psnr = val_psnr / len(val_loader)
            scheduler.step()

            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                        f"{avg_train_psnr:.2f},{avg_val_psnr:.2f}\n")

            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Train PSNR: {avg_train_psnr:.2f} | "
                  f"Val PSNR: {avg_val_psnr:.2f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Early stopping check
            if avg_val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), config.checkpoint_dir / "best_model.pth")
                print(f"Saved new best model with val loss {avg_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement}/{config.early_stopping_patience} epochs")
                
                if epochs_without_improvement >= config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

            if (epoch + 1) % 5 == 0:
                ckpt_path = config.checkpoint_dir / f"model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint at epoch {epoch+1}")
            if (epoch + 1) % 5 == 0:
                #visualize_forward_process(config, diffusion, epoch+1)
                model.eval()
                with torch.no_grad():
                    for x0, context in val_loader:
                        x0 = x0.to(device)
                        context = context.to(device)
                        x0_single = x0[0:1]
                        context_single = context[0:1]

                        sqrt_alpha_bar = torch.sqrt(config.alpha_bar[1999]).view(-1, 1, 1, 1)
                        sqrt_one_minus_alpha_bar = torch.sqrt(1 - config.alpha_bar[1999]).view(-1, 1, 1, 1)
                        noise = torch.randn_like(x0_single)
                        xt = sqrt_alpha_bar * x0_single + sqrt_one_minus_alpha_bar * noise

                        pred_noise, pred_x0_direct = model(xt, torch.tensor([1999], device=device).repeat(1), context_single)
                        alpha_bar_t = config.alpha_bar[1999].view(-1, 1, 1, 1)
                        pred_x0 = (xt - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
                        pred_x0_original = pred_x0 * config.image_std.view(1, -1, 1, 1) + config.image_mean.view(1, -1, 1, 1)
                        pred_x0_original = pred_x0_original[:, :, :config.img_dims[0], :config.img_dims[1]]

                        random_noise = torch.randn_like(x0_single)[:, :, :config.img_dims[0], :config.img_dims[1]]

                        generated = generate_samples(model, config, num_samples=1, context=context_single)
                        generated = generated[:, :, :config.img_dims[0], :config.img_dims[1]]

                        vis_dir = config.output_dir / "visualizations" / f"epoch_{epoch+1}"
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        for band_idx in range(generated.shape[1]):
                            fig, axs = plt.subplots(1, 3, figsize=(25, 5))

                            orig_band = x0_single[0, band_idx].cpu().numpy()
                            axs[0].imshow(orig_band, cmap='gray')
                            axs[0].set_title(f"Original Band {band_idx+1}")
                            axs[0].axis('off')

                            random_noise_band = random_noise[0, band_idx].cpu().numpy()
                            axs[1].imshow(random_noise_band, cmap='gray')
                            axs[1].set_title(f"Random Noise Band {band_idx+1}")
                            axs[1].axis('off')

                            gen_band = generated[0, band_idx].cpu().numpy()
                            axs[2].imshow(gen_band, cmap='gray')
                            axs[2].set_title(f"Generated Band {band_idx+1}")
                            axs[2].axis('off')

                            plt.tight_layout()
                            plt.savefig(vis_dir / f"epoch_{epoch+1}_band_{band_idx+1}.png")
                            plt.close()

                        break

        plot_loss_curves(log_file, config.output_dir)

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current state...")
        ckpt_path = config.checkpoint_dir / "interrupted_model.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved intermediate model to {ckpt_path}")

    return model

#%% --------------- IMPROVED GENERATION ---------------
def generate_samples(model, config, num_samples=1, context=None, num_steps=None, use_ddim=True):
    model.eval()
    diffusion = DiffusionProcess(config)
    num_steps = num_steps or config.T

    step_sequence = get_timestep_sequence(num_steps, config.T)

    if context is None:
        context = torch.randn((num_samples, config.context_dims[0], *config.padded_dims), device=device)

    x_t = torch.randn((num_samples, config.num_bands, *config.padded_dims), device=device)

    for t in tqdm(reversed(step_sequence), desc="Generating", total=len(step_sequence)):
        pred_noise, _ = model(x_t, torch.full((num_samples,), t, device=device, dtype=torch.long), context)
        x_t = diffusion.reverse_diffusion_step(
            model, x_t, t, context, use_ddim=use_ddim
        )

    generated_diff = x_t[:, :, :config.img_dims[0], :config.img_dims[1]]
    generated_diff = generated_diff * config.image_std.view(1, -1, 1, 1) + config.image_mean.view(1, -1, 1, 1)

    return generated_diff.cpu()

def get_timestep_sequence(num_steps, total_timesteps):
    return np.linspace(0, total_timesteps-1, num_steps).astype(int)[::-1].tolist()

#%% --------------- GENERATION AND VISUALIZATION ---------------
def save_band_png(data, path):
    data = data.cpu().numpy()
    if len(data.shape) == 3:
        data = data[0]
    if data.ndim == 1:
        side = int(np.sqrt(len(data)))
        data = data.reshape(side, side)

    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)
    plt.imsave(path, data, cmap='gray', vmin=vmin, vmax=vmax)

def save_tiff(tensor, path):
    tensor = tensor.cpu().numpy()
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=tensor.shape[1],
        width=tensor.shape[2],
        count=tensor.shape[0],
        dtype=tensor.dtype
    ) as dst:
        for i in range(tensor.shape[0]):
            dst.write(tensor[i], i + 1)

def test_model(config, model_path, context_path, num_samples=1):
    try:
        model = create_diffusion_model(config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")

        target_context = torch.load(context_path, map_location=device)

        context_diff = target_context
        if context_diff.dim() == 3:
            context_diff = context_diff.unsqueeze(0)

        context_diff = (context_diff - config.context_mean.to(device)) / config.context_std.to(device)
        context_diff = F.pad(context_diff, (0, config.pad_w, 0, config.pad_h))

        print("Starting image generation with improved DDIM...")
        generated = generate_samples(
            model,
            config,
            num_samples=num_samples,
            context=context_diff,
            num_steps=250,
            use_ddim=True
        )

        test_dir = config.output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        for i in range(num_samples):
            save_tiff(generated[i], test_dir / f"generated_{i}.tif")
        print(f"Saved {num_samples} generated images to {test_dir}")
    except Exception as e:
        print(f"Error in test_model: {str(e)}")
        import traceback
        traceback.print_exc()

#%% --------------- MAIN EXECUTION ---------------
if __name__ == "__main__":
    try:
        data_root = Path(path_data)
        data_specs = analyze_data_structure(data_root)
        config = DynamicConfig(data_specs)

        print("\nTraining Configuration:")
        print(f"Number of sequences: {config.num_sequences}")
        print(f"Image dimensions: {config.img_dims}")
        print(f"Number of bands: {config.num_bands}")
        print(f"Context dimensions: {config.context_dims}")
        print(f"\nTraining Parameters:")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.epochs}")
        print(f"Base channels: {config.base_channels}")

        model = train_model(config)

        print("Loading best model for generation...")
        best_model_path = config.checkpoint_dir / "best_model.pth"
        if not best_model_path.exists():
            print(f"Best model not found at {best_model_path}, using the trained model")
        else:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print("Best model loaded successfully")

    except Exception as e:
        print(f"\nCritical error occurred: {str(e)}")
        import traceback
        traceback.print_exc()