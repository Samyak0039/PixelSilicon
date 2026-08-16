import os
import sys
import csv
import time
from typing import Tuple

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import create_train_val_datasets
from src.model import PixelSiliconNet, print_trainable_parameters


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Differentiable L1 Variant).
    Formula: L = sqrt((prediction - target)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred.float() - target.float()
        return torch.mean(torch.sqrt(diff * diff + self.eps_sq))


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) Loss for 1-channel Grayscale Images.
    Computes Loss = 1.0 - SSIM(pred, target).
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.channel = 1

        # Create 1D Gaussian kernel
        gauss = torch.exp(-torch.arange(window_size).float().sub(window_size // 2).pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()

        # Create 2D Gaussian window
        _1d_window = gauss.unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _2d_window)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1 = img1.float()
        img2 = img2.float()

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(device=img1.device, dtype=torch.float32)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


def compute_psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Computes mean Peak Signal-to-Noise Ratio (PSNR) over a batch of images.
    """
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10.0 * torch.log10((max_val ** 2) / mse)
    return float(psnr.mean().item())


def save_visual_comparison(noisy_lr: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, save_path: str):
    """
    Saves a 3-panel visual comparison:
    1. NoisyLR bicubic-upscaled to 256x256
    2. PixelSilicon prediction
    3. Ground Truth
    """
    noisy_np = noisy_lr.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()

    # Bicubic upscale NoisyLR from 128x128 to 256x256
    bicubic_np = cv2.resize(noisy_np, (256, 256), interpolation=cv2.INTER_CUBIC)
    bicubic_clamped = np.clip(bicubic_np, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    im0 = axes[0].imshow(bicubic_clamped, cmap='gray')
    axes[0].set_title("1. NoisyLR Bicubic Upscaled\n(256x256)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_np, cmap='gray')
    axes[1].set_title("2. PixelSilicon Prediction\n(256x256)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(gt_np, cmap='gray')
    axes[2].set_title("3. Ground Truth (GT)\n(256x256)", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Stable Training Best Model Comparison", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def train_pixelsilicon_stable():
    # Directories setup
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Checkpoint & output paths (Specifically non-destructive to models/pixelsilicon_best.pth)
    best_model_path = os.path.join(models_dir, "pixelsilicon_best_stable.pth")
    final_model_path = os.path.join(models_dir, "pixelsilicon_final_stable.pth")
    csv_history_path = os.path.join(results_dir, "training_history_stable.csv")
    train_loss_plot_path = os.path.join(results_dir, "training_loss_stable.png")
    val_metrics_plot_path = os.path.join(results_dir, "validation_metrics_stable.png")
    best_vis_comparison_path = os.path.join(results_dir, "best_validation_comparison_stable.png")

    # Hyperparameters & Configurations for Stable Training Run
    batch_size = 4
    max_epochs = 20
    learning_rate = 3e-5
    grad_clip_max_norm = 1.0
    seed = 42

    scheduler_factor = 0.5
    scheduler_patience = 2
    scheduler_min_lr = 1e-6

    early_stopping_patience = 5

    # 1. Hardware & System Setup (No AMP, standard float32)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # Print Complete Startup Configuration
    print("=" * 70, flush=True)
    print("        STABLE SECOND TRAINING RUN CONFIGURATION SUMMARY        ", flush=True)
    print("=" * 70, flush=True)
    print(f"CUDA Available:           {cuda_available}", flush=True)
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Name:                 {gpu_name}", flush=True)
        print(f"Total VRAM:               {total_vram_gb:.2f} GB", flush=True)
    print(f"Execution Precision:      Standard float32 (AMP Disabled)", flush=True)
    print(f"Batch Size:               {batch_size}", flush=True)
    print(f"Maximum Epochs:           {max_epochs}", flush=True)
    print(f"Initial Learning Rate:    {learning_rate:.2e}", flush=True)
    print(f"Optimizer:                Adam", flush=True)
    print(f"Gradient Clipping:        max_norm = {grad_clip_max_norm}", flush=True)
    print(f"Loss Function:            Charbonnier + 0.1 * SSIM", flush=True)
    print(f"Dataset Split:            80% Train (2560) / 20% Val (640), Seed = {seed}", flush=True)
    print(f"LR Scheduler:             ReduceLROnPlateau (mode='max', factor={scheduler_factor}, patience={scheduler_patience}, min_lr={scheduler_min_lr})", flush=True)
    print(f"Early Stopping:           patience = {early_stopping_patience} (based on Val PSNR)", flush=True)
    print(f"Target Best Checkpoint:   {best_model_path}", flush=True)
    print(f"Target Final Checkpoint:  {final_model_path}", flush=True)
    print(f"Protected Checkpoint:     {os.path.join(models_dir, 'pixelsilicon_best.pth')} (PRESERVED)", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Load Datasets & DataLoaders
    train_ds, val_ds = create_train_val_datasets(
        root_dir=os.path.join(project_root, "dataset", "KLA", "train", "train"),
        val_ratio=0.20,
        seed=seed,
        normalize_gt=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=cuda_available)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=cuda_available)

    # Instantiate Model
    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2).to(device)
    print("\n--- Model Architecture ---", flush=True)
    print_trainable_parameters(model)

    # Optimizer, Scheduler & Losses
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )

    charbonnier_fn = CharbonnierLoss(eps=1e-6).to(device)
    ssim_fn = SSIMLoss().to(device)

    def calc_total_loss(pred, target):
        loss_c = charbonnier_fn(pred, target)
        loss_s = ssim_fn(pred, target)
        return loss_c + 0.1 * loss_s

    # Initial one-batch forward pass verification
    print("=" * 70, flush=True)
    print("             INITIAL ONE-BATCH FORWARD PASS CHECK            ", flush=True)
    print("=" * 70, flush=True)
    sample_input, sample_target = next(iter(train_loader))
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)

    model.eval()
    with torch.no_grad():
        sample_pred = model(sample_input)
        sample_loss = calc_total_loss(sample_pred, sample_target)

    print(f"Input Shape:        {list(sample_input.shape)}", flush=True)
    print(f"Target Shape:       {list(sample_target.shape)}", flush=True)
    print(f"Prediction Shape:   {list(sample_pred.shape)}", flush=True)
    print(f"Initial Loss:       {sample_loss.item():.6f}", flush=True)
    print("=" * 70 + "\n", flush=True)

    history = []
    best_val_psnr = -1.0
    epochs_no_improve = 0

    print(f"Beginning Stable Training Loop (Max {max_epochs} Epochs)...", flush=True)
    print("-" * 70, flush=True)

    start_train_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = calc_total_loss(preds, targets)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_ds)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_psnr = 0.0
        running_val_ssim = 0.0

        first_val_batch = None

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                preds = model(inputs)
                loss = calc_total_loss(preds, targets)

                running_val_loss += loss.item() * inputs.size(0)

                # Compute PSNR & SSIM metrics (standard float32)
                psnr_val = compute_psnr_batch(preds, targets)
                ssim_val = 1.0 - ssim_fn(preds, targets).item()

                running_val_psnr += psnr_val * inputs.size(0)
                running_val_ssim += ssim_val * inputs.size(0)

                if first_val_batch is None:
                    first_val_batch = (inputs[0].detach(), preds[0].detach(), targets[0].detach())

        epoch_val_loss = running_val_loss / len(val_ds)
        epoch_val_psnr = running_val_psnr / len(val_ds)
        epoch_val_ssim = running_val_ssim / len(val_ds)

        # Update ReduceLROnPlateau scheduler based on Validation PSNR
        scheduler.step(epoch_val_psnr)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_duration = time.time() - epoch_start_time

        # Check for improvement in validation PSNR
        is_best = epoch_val_psnr > best_val_psnr
        best_str = ""
        if is_best:
            best_val_psnr = epoch_val_psnr
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            best_str = " [BEST SAVED]"

            # Save best visual comparison plot
            if first_val_batch is not None:
                save_visual_comparison(
                    noisy_lr=first_val_batch[0],
                    pred=first_val_batch[1],
                    gt=first_val_batch[2],
                    save_path=best_vis_comparison_path
                )
        else:
            epochs_no_improve += 1

        # Record metrics history
        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "val_psnr": epoch_val_psnr,
            "val_ssim": epoch_val_ssim,
            "learning_rate": current_lr
        })

        # Print concise epoch metrics: Epoch, Train Loss, Val Loss, Val PSNR, Val SSIM, Current LR
        print(f"Epoch [{epoch:02d}/{max_epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f} | "
              f"Val PSNR: {epoch_val_psnr:.4f} dB | "
              f"Val SSIM: {epoch_val_ssim:.4f} | "
              f"LR: {current_lr:.2e}{best_str}", flush=True)
        sys.stdout.flush()

        # Early Stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (No improvement in Val PSNR for {early_stopping_patience} consecutive epochs).", flush=True)
            break

    total_time = time.time() - start_train_time
    print("-" * 70, flush=True)
    print(f"Stable Training Complete in {total_time / 60:.2f} minutes.", flush=True)
    print(f"Best Validation PSNR: {best_val_psnr:.4f} dB", flush=True)

    # Save final model checkpoint
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}", flush=True)
    print(f"Best model saved to:  {best_model_path}", flush=True)

    # Save training history CSV
    with open(csv_history_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "learning_rate"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to: {csv_history_path}", flush=True)

    # Generate training loss plot
    epochs_arr = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_psnrs = [h["val_psnr"] for h in history]
    val_ssims = [h["val_ssim"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_arr, train_losses, label="Train Loss", color="blue", linewidth=2)
    plt.plot(epochs_arr, val_losses, label="Validation Loss", color="red", linewidth=2)
    plt.title("PixelSiliconNet Stable Training & Validation Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Charbonnier + 0.1 * SSIM)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(train_loss_plot_path, dpi=300)
    plt.close()
    print(f"Training loss plot saved to: {train_loss_plot_path}", flush=True)

    # Generate validation metrics plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:green"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation PSNR (dB)", color=color, fontweight='bold')
    line1 = ax1.plot(epochs_arr, val_psnrs, color=color, linewidth=2, label="Val PSNR (dB)")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Validation SSIM", color=color, fontweight='bold')
    line2 = ax2.plot(epochs_arr, val_ssims, color=color, linewidth=2, linestyle="--", label="Val SSIM")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    plt.title("PixelSiliconNet Stable Validation PSNR & SSIM", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(val_metrics_plot_path, dpi=300)
    plt.close()
    print(f"Validation metrics plot saved to: {val_metrics_plot_path}", flush=True)


if __name__ == "__main__":
    train_pixelsilicon_stable()
