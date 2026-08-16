import os
import sys
import csv
import time
from typing import Tuple

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

from src.dataset import create_train_val_datasets
from src.model import PixelSiliconNet


def evaluate_pixelsilicon_val():
    """
    Evaluates the trained PixelSiliconNet model on the KLA 640-image validation split.
    Calculates PSNR, SSIM, and LPIPS metrics directly from the validation images.
    Saves outputs to results/pixelsilicon_validation_metrics.txt and results/pixelsilicon_validation_metrics.csv.
    """
    # Setup paths
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "pixelsilicon_best_stable.pth")
    txt_output_path = os.path.join(results_dir, "pixelsilicon_validation_metrics.txt")
    csv_output_path = os.path.join(results_dir, "pixelsilicon_validation_metrics.csv")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # 5. Hardware inspection & GPU detection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    # 1 & 2. Load validation dataset using identical deterministic split parameters (80/20, seed=42)
    _, val_dataset = create_train_val_datasets(
        root_dir=os.path.join(project_root, "dataset", "KLA", "train", "train"),
        val_ratio=0.20,
        seed=42,
        normalize_gt=True
    )
    num_val_images = len(val_dataset)

    # 3 & 4. Instantiate PixelSiliconNet architecture and load best stable checkpoint weights
    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # 6. Set model to evaluation mode
    model.eval()

    # Calculate model parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load LPIPS perceptual metric network (AlexNet backbone, net='alex')
    # Used purely as an evaluation loss metric, not as model backbone weights.
    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_fn.eval()

    # Batch processing with safe batch size for GTX 1060 (6GB VRAM)
    batch_size = 8
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Warm up GPU memory allocator for precise timing measurement
    dummy_input = torch.zeros(1, 1, 128, 128, device=device)
    with torch.inference_mode():
        _ = model(dummy_input)
    if cuda_available:
        torch.cuda.synchronize()

    print("=================================================================", flush=True)
    print(f"Beginning Evaluation on {num_val_images} Validation Images...", flush=True)
    print("=================================================================", flush=True)

    start_time = time.time()
    processed_count = 0

    # 7. Use torch.inference_mode() during evaluation
    with torch.inference_mode():
        for b_idx, (noisy_lr_batch, gt_batch) in enumerate(val_loader):
            noisy_lr_batch = noisy_lr_batch.to(device)
            gt_batch = gt_batch.to(device)

            # Generate PixelSiliconNet predictions [B, 1, 256, 256]
            pred_batch = model(noisy_lr_batch)

            # 4. Ensure predictions and GT targets are float32 and clamped to [0.0, 1.0]
            pred_clamped = torch.clamp(pred_batch, 0.0, 1.0)
            gt_clamped = torch.clamp(gt_batch, 0.0, 1.0)

            # LPIPS Requirements:
            # - LPIPS expects 3 channels: repeat grayscale channel 3 times [B, 3, 256, 256]
            # - Convert range [0, 1] to [-1, 1] before passing to LPIPS
            pred_rgb = pred_clamped.repeat(1, 3, 1, 1) * 2.0 - 1.0
            gt_rgb = gt_clamped.repeat(1, 3, 1, 1) * 2.0 - 1.0

            # Compute LPIPS for batch [B]
            lpips_batch = lpips_fn(pred_rgb, gt_rgb).view(-1).cpu().numpy()
            lpips_values.extend(lpips_batch.tolist())

            # Convert predictions and GT targets to CPU NumPy arrays for PSNR & SSIM
            pred_np_batch = pred_clamped.squeeze(1).cpu().numpy()  # [B, 256, 256]
            gt_np_batch = gt_clamped.squeeze(1).cpu().numpy()      # [B, 256, 256]

            current_batch_size = pred_np_batch.shape[0]

            for i in range(current_batch_size):
                pred_img = pred_np_batch[i]
                gt_img = gt_np_batch[i]

                # 5. Compute PSNR (Peak Signal-to-Noise Ratio)
                psnr_val = compute_psnr(gt_img, pred_img, data_range=1.0)
                psnr_values.append(float(psnr_val))

                # 6. Compute SSIM (Structural Similarity Index)
                ssim_val = compute_ssim(gt_img, pred_img, data_range=1.0)
                ssim_values.append(float(ssim_val))

            processed_count += current_batch_size

            # Print progress periodically (every 100 images)
            if processed_count % 100 == 0 or processed_count == num_val_images:
                print(f"Evaluated {processed_count:03d} / {num_val_images} validation samples...", flush=True)

    if cuda_available:
        torch.cuda.synchronize()

    total_eval_time = time.time() - start_time
    avg_time_per_image_ms = (total_eval_time / num_val_images) * 1000.0

    # Calculate average validation metrics across all 640 validation images
    pixelsilicon_psnr = float(np.mean(psnr_values))
    pixelsilicon_ssim = float(np.mean(ssim_values))
    pixelsilicon_lpips = float(np.mean(lpips_values))

    # Bicubic baseline metrics for comparison
    bicubic_psnr = 22.9449
    bicubic_ssim = 0.5326
    bicubic_lpips = 0.4502

    # Calculate improvements (positive means PixelSilicon is better)
    psnr_imp = pixelsilicon_psnr - bicubic_psnr
    ssim_imp = pixelsilicon_ssim - bicubic_ssim
    lpips_imp = bicubic_lpips - pixelsilicon_lpips  # Lower LPIPS distance is better

    # Format requested final text report
    txt_report = f"""============================================================
PIXELSILICON VALIDATION EVALUATION
============================================================
Validation Images:                     {num_val_images}
GPU Name:                              {gpu_name}
Model Parameter Count:                 {total_params:,} ({total_params / 1e6:.4f} M)
Total Evaluation Time:                 {total_eval_time:.4f} s
Average Evaluation Time per Image:     {avg_time_per_image_ms:.2f} ms
------------------------------------------------------------
PSNR:                                  {pixelsilicon_psnr:.4f} dB
SSIM:                                  {pixelsilicon_ssim:.4f}
LPIPS:                                 {pixelsilicon_lpips:.4f}
------------------------------------------------------------
PSNR Improvement over Bicubic:         +{psnr_imp:.4f} dB
SSIM Improvement over Bicubic:         +{ssim_imp:.4f}
LPIPS Improvement over Bicubic:        +{lpips_imp:.4f}
============================================================
"""

    print("\n" + txt_report, flush=True)

    # Save final results to results/pixelsilicon_validation_metrics.txt
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(txt_report + "\n")
    print(f"Validation metrics report saved to: {txt_output_path}", flush=True)

    # Save CSV file to results/pixelsilicon_validation_metrics.csv
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "psnr", "ssim", "lpips"])
        writer.writerow(["Bicubic", f"{bicubic_psnr:.4f}", f"{bicubic_ssim:.4f}", f"{bicubic_lpips:.4f}"])
        writer.writerow(["PixelSilicon", f"{pixelsilicon_psnr:.4f}", f"{pixelsilicon_ssim:.4f}", f"{pixelsilicon_lpips:.4f}"])

    print(f"Validation metrics CSV saved to:    {csv_output_path}", flush=True)


if __name__ == "__main__":
    evaluate_pixelsilicon_val()
