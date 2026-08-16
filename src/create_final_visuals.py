import os
import sys
import time

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.dataset import create_train_val_datasets
from src.model import PixelSiliconNet


def compute_sobel_gradient(img: np.ndarray) -> np.ndarray:
    """
    Computes 2D Sobel gradient magnitude for edge visualization.
    """
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def generate_final_visuals():
    """
    Generates publication-quality visual evidence figures for the PixelSilicon project.
    
    Figures Created:
    1. results/final_visual_comparison.png:
       4 representative validation samples showing Full Images (with bounding box) 
       and Zoomed-in Detail Crops across:
       Col 1: Original NoisyLR (upscaled to 256x256)
       Col 2: Bicubic Interpolation
       Col 3: PixelSiliconNet Restoration
       Col 4: Ground Truth (GT)
    
    2. results/final_edge_comparison.png:
       Sobel gradient magnitude edge maps for Bicubic vs PixelSilicon vs Ground Truth.
    """
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "pixelsilicon_best_stable.pth")
    fig1_path = os.path.join(results_dir, "final_visual_comparison.png")
    fig2_path = os.path.join(results_dir, "final_edge_comparison.png")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Hardware & GPU inspection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # Load 640 validation images using deterministic split (80/20, seed=42)
    _, val_dataset = create_train_val_datasets(
        root_dir=os.path.join(project_root, "dataset", "KLA", "train", "train"),
        val_ratio=0.20,
        seed=42,
        normalize_gt=True
    )

    # Load PixelSiliconNet model & best stable checkpoint weights
    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Select 4 representative validation samples with varied structural content
    sample_indices = [0, 40, 120, 250]

    # Crop box coordinates (y_min, x_min, crop_size=64) containing fine structures
    crop_boxes = [
        (80, 80, 64),   # Sample 0
        (90, 90, 64),   # Sample 40
        (70, 70, 64),   # Sample 120
        (100, 100, 64)  # Sample 250
    ]

    samples_data = []

    print("Processing visual comparison samples...", flush=True)

    with torch.inference_mode():
        for idx_idx, val_idx in enumerate(sample_indices):
            noisy_lr_tensor, gt_tensor = val_dataset[val_idx]

            dataset_idx = val_dataset.indices[val_idx]
            filename = val_dataset.dataset.filenames[dataset_idx]

            noisy_lr_t = noisy_lr_tensor.unsqueeze(0).to(device)
            pred_t = model(noisy_lr_t)

            noisy_lr_np = noisy_lr_tensor.squeeze().cpu().numpy()  # (128, 128)
            pred_np = torch.clamp(pred_t, 0.0, 1.0).squeeze().cpu().numpy()  # (256, 256)
            gt_np = torch.clamp(gt_tensor, 0.0, 1.0).squeeze().cpu().numpy() # (256, 256)

            # Bicubic 256x256 interpolation
            bicubic_np = cv2.resize(noisy_lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)
            bicubic_clamped = np.clip(bicubic_np, 0.0, 1.0)

            # Original NoisyLR upscaled to 256x256 (nearest neighbor to show raw sensor pixels)
            noisy_raw_upscaled = cv2.resize(noisy_lr_np, (256, 256), interpolation=cv2.INTER_NEAREST)
            noisy_raw_clamped = np.clip(noisy_raw_upscaled, 0.0, 1.0)

            # Sobel Edge Magnitude Maps
            grad_bicubic = compute_sobel_gradient(bicubic_clamped)
            grad_pred = compute_sobel_gradient(pred_np)
            grad_gt = compute_sobel_gradient(gt_np)

            samples_data.append({
                "filename": filename,
                "noisy_raw": noisy_raw_clamped,
                "bicubic": bicubic_clamped,
                "pixelsilicon": pred_np,
                "gt": gt_np,
                "grad_bicubic": grad_bicubic,
                "grad_pixelsilicon": grad_pred,
                "grad_gt": grad_gt,
                "crop": crop_boxes[idx_idx]
            })

    # --- FIGURE 1: Publication-Quality Image Restoration Comparison ---
    fig1, axes1 = plt.subplots(8, 4, figsize=(16, 26))

    col_titles = [
        "Col 1: Original KLA NoisyLR (256x256)",
        "Col 2: Bicubic Reconstruction",
        "Col 3: PixelSiliconNet Restoration",
        "Col 4: Ground Truth (GT)"
    ]

    for c in range(4):
        axes1[0, c].set_title(col_titles[c], fontsize=13, fontweight="bold", pad=12)

    for i, sdata in enumerate(samples_data):
        r_full = i * 2
        r_crop = i * 2 + 1

        y_min, x_min, crop_sz = sdata["crop"]
        imgs = [sdata["noisy_raw"], sdata["bicubic"], sdata["pixelsilicon"], sdata["gt"]]

        for c, img in enumerate(imgs):
            # Full Image Panel
            ax_f = axes1[r_full, c]
            ax_f.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            rect = patches.Rectangle((x_min, y_min), crop_sz, crop_sz, linewidth=2, edgecolor="red", facecolor="none")
            ax_f.add_patch(rect)
            if c == 0:
                ax_f.set_ylabel(f"Sample {sdata['filename']}\nFull Image", fontsize=11, fontweight="bold")
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            # Zoomed Detail Crop Panel
            crop_img = img[y_min:y_min + crop_sz, x_min:x_min + crop_sz]
            ax_c = axes1[r_crop, c]
            ax_c.imshow(crop_img, cmap="gray", vmin=0.0, vmax=1.0)
            for spine in ax_c.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
            if c == 0:
                ax_c.set_ylabel("Zoomed Detail Crop", fontsize=11, fontweight="bold")
            ax_c.set_xticks([])
            ax_c.set_yticks([])

    plt.suptitle("PixelSiliconNet Semiconductor Wafer Image Restoration & 2x Super-Resolution", fontsize=16, fontweight="bold", y=0.996)
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Publication-quality visual comparison saved to: {fig1_path}", flush=True)

    # --- FIGURE 2: Sobel Edge Map Comparison ---
    fig2, axes2 = plt.subplots(4, 3, figsize=(14, 16))

    edge_col_titles = [
        "Bicubic Edge Map",
        "PixelSilicon Edge Map",
        "Ground Truth Edge Map"
    ]

    for c in range(3):
        axes2[0, c].set_title(edge_col_titles[c], fontsize=13, fontweight="bold", pad=12)

    for i, sdata in enumerate(samples_data):
        edge_maps = [sdata["grad_bicubic"], sdata["grad_pixelsilicon"], sdata["grad_gt"]]

        for c, emap in enumerate(edge_maps):
            ax = axes2[i, c]
            im = ax.imshow(emap, cmap="inferno")
            if c == 0:
                ax.set_ylabel(f"Sample {sdata['filename']}", fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Structural Edge & High-Frequency Detail Comparison (Sobel Gradient Magnitude)", fontsize=16, fontweight="bold", y=0.996)
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Edge map comparison figure saved to: {fig2_path}", flush=True)


if __name__ == "__main__":
    generate_final_visuals()
