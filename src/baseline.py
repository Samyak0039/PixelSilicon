import os
import sys

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

from src.dataset import create_train_val_datasets


def calculate_bicubic_baseline():
    """
    Evaluates the Bicubic Interpolation Baseline on the KLA Validation Dataset.

    Steps:
    1. Loads the validation split (80/20 split, seed=42).
    2. Upscales NoisyLR (128x128) to 256x256 using OpenCV bicubic interpolation.
    3. Clamps upscaled values to [0, 1] range matching normalized Ground Truth targets.
    4. Calculates PSNR, SSIM, and LPIPS metrics across all validation images.
    5. Saves metrics to results/baseline_metrics.txt and visual comparison to results/bicubic_baseline.png.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bicubic Baseline Evaluation running on device: {device}")

    # Output directory setup
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, "baseline_metrics.txt")
    plot_path = os.path.join(results_dir, "bicubic_baseline.png")

    # 1. Load validation dataset using identical deterministic split parameters (80/20, seed=42)
    _, val_dataset = create_train_val_datasets(
        root_dir=os.path.join(project_root, "dataset", "KLA", "train", "train"),
        val_ratio=0.20,
        seed=42,
        normalize_gt=True
    )

    num_val_images = len(val_dataset)
    print(f"Validation dataset loaded successfully ({num_val_images} samples).")

    # Load LPIPS perceptual loss metric (AlexNet backbone)
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_fn.eval()

    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Store validation sample for visual comparison (000000.npy if present, else first sample)
    sample_vis_data = None
    target_filename = "000000.npy"

    print("Processing validation samples...")
    for idx in range(num_val_images):
        # Retrieve sample pair from dataset
        noisy_lr_tensor, gt_tensor = val_dataset[idx]

        # Extract filename from underlying dataset instance
        sample_idx = val_dataset.indices[idx]
        filename = val_dataset.dataset.filenames[sample_idx]

        # Convert tensors to 2D numpy arrays for OpenCV processing
        noisy_lr_np = noisy_lr_tensor.squeeze().cpu().numpy()  # Shape: (128, 128)
        gt_np = gt_tensor.squeeze().cpu().numpy()              # Shape: (256, 256)

        # 2. Upscale NoisyLR from 128x128 to 256x256 using bicubic interpolation
        bicubic_np = cv2.resize(noisy_lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)

        # 4. Clamp bicubic prediction and GT target to [0.0, 1.0] for consistent evaluation
        bicubic_clamped = np.clip(bicubic_np, 0.0, 1.0)
        gt_clamped = np.clip(gt_np, 0.0, 1.0)

        # Compute PSNR (Peak Signal-to-Noise Ratio)
        psnr_val = compute_psnr(gt_clamped, bicubic_clamped, data_range=1.0)
        psnr_values.append(psnr_val)

        # Compute SSIM (Structural Similarity Index)
        ssim_val = compute_ssim(gt_clamped, bicubic_clamped, data_range=1.0)
        ssim_values.append(ssim_val)

        # 5 & 6. Compute LPIPS (Learned Perceptual Image Patch Similarity)
        # Prepare 4D PyTorch tensor [1, 1, 256, 256]
        bicubic_t = torch.from_numpy(bicubic_clamped).float().unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt_clamped).float().unsqueeze(0).unsqueeze(0)

        # Repeat 3 channels for RGB input format & normalize range [0, 1] -> [-1, 1]
        bicubic_rgb = bicubic_t.repeat(1, 3, 1, 1) * 2.0 - 1.0
        gt_rgb = gt_t.repeat(1, 3, 1, 1) * 2.0 - 1.0

        bicubic_rgb = bicubic_rgb.to(device)
        gt_rgb = gt_rgb.to(device)

        with torch.no_grad():
            lpips_val = lpips_fn(bicubic_rgb, gt_rgb).item()
        lpips_values.append(lpips_val)

        # Record visual comparison sample (prefer 000000.npy if in validation split)
        if sample_vis_data is None or filename == target_filename:
            sample_vis_data = {
                "filename": filename,
                "noisy_lr": noisy_lr_np,
                "bicubic": bicubic_clamped,
                "gt": gt_clamped
            }

    # 3. Calculate mean dataset metrics
    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    avg_lpips = float(np.mean(lpips_values))

    # 8. Print dataset baseline results
    print("\n" + "=" * 55)
    print("        BICUBIC INTERPOLATION BASELINE RESULTS        ")
    print("=" * 55)
    print(f"Validation Images Evaluated: {num_val_images}")
    print(f"Average PSNR:                {avg_psnr:.4f} dB")
    print(f"Average SSIM:                {avg_ssim:.4f}")
    print(f"Average LPIPS:               {avg_lpips:.4f}")
    print("=" * 55 + "\n")

    # 9. Save metrics to results/baseline_metrics.txt
    with open(metrics_file_path, "w") as f:
        f.write("BICUBIC BASELINE METRICS (KLA Validation Split)\n")
        f.write("================================================\n")
        f.write(f"Validation Images: {num_val_images}\n")
        f.write(f"Average PSNR:      {avg_psnr:.4f} dB\n")
        f.write(f"Average SSIM:      {avg_ssim:.4f}\n")
        f.write(f"Average LPIPS:     {avg_lpips:.4f}\n")

    print(f"Baseline metrics successfully written to: {metrics_file_path}")

    # 10. Save visual comparison plot to results/bicubic_baseline.png
    if sample_vis_data is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

        # Panel 1: Original NoisyLR (128x128)
        im0 = axes[0].imshow(sample_vis_data["noisy_lr"], cmap='gray')
        axes[0].set_title(f"1. NoisyLR (128x128)\nSample: {sample_vis_data['filename']}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: Bicubic 256x256
        im1 = axes[1].imshow(sample_vis_data["bicubic"], cmap='gray')
        axes[1].set_title("2. Bicubic 256x256", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Ground Truth GT 256x256
        im2 = axes[2].imshow(sample_vis_data["gt"], cmap='gray')
        axes[2].set_title("3. Ground Truth GT (256x256)", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(f"Bicubic Baseline Comparison - Sample {sample_vis_data['filename']}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visual comparison plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    calculate_bicubic_baseline()
