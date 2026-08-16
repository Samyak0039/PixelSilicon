import os
import sys
import csv
import time
from typing import Tuple, List, Dict

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import create_train_val_datasets
from src.model import PixelSiliconNet


def compute_sobel_gradient(img: np.ndarray) -> np.ndarray:
    """
    Computes 2D Sobel gradient magnitude for a 2D float32 grayscale image array in [0.0, 1.0].
    
    NOTE ON EVALUATION METHODOLOGY:
    This function extracts high-frequency structural gradient magnitude maps to quantify 
    fine edge and defect detail preservation. This is a deterministic structural/edge 
    preservation metric, NOT a defect classification or defect detection model.
    """
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return grad_mag


def compute_edge_metrics(grad_rec: np.ndarray, grad_gt: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculates structural edge and gradient metrics between reconstruction and GT:
    1. Pearson Gradient Correlation
    2. Adaptive Edge Precision
    3. Adaptive Edge Recall
    4. Edge F1-Score
    5. Mean Gradient Magnitude Ratio
    """
    # 1. Pearson correlation between gradient magnitude maps
    rec_flat = grad_rec.ravel()
    gt_flat = grad_gt.ravel()
    std_rec = np.std(rec_flat)
    std_gt = np.std(gt_flat)

    if std_rec < 1e-8 or std_gt < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(rec_flat, gt_flat)[0, 1])

    # 2. Adaptive Edge Thresholding based on 85th percentile of GT gradient magnitude
    thresh = float(np.percentile(grad_gt, 85))
    edge_gt = (grad_gt >= thresh)
    edge_rec = (grad_rec >= thresh)

    tp = np.sum(edge_gt & edge_rec)
    fp = np.sum((~edge_gt) & edge_rec)
    fn = np.sum(edge_gt & (~edge_rec))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 3. Average Gradient Magnitude Ratio (Reconstruction / GT)
    mean_gt = float(np.mean(grad_gt))
    mean_rec = float(np.mean(grad_rec))
    grad_ratio = float(mean_rec / mean_gt) if mean_gt > 1e-8 else 0.0

    return corr, precision, recall, f1, grad_ratio


def evaluate_defect_preservation():
    """
    Evaluates fine structural and edge preservation performance on the 640-sample KLA validation split.
    Compares PixelSiliconNet restoration against standard Bicubic interpolation baseline.
    Outputs metrics to text and CSV format, and generates visual gradient/edge maps.
    """
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "pixelsilicon_best_stable.pth")
    txt_output_path = os.path.join(results_dir, "defect_preservation_metrics.txt")
    csv_output_path = os.path.join(results_dir, "defect_preservation_metrics.csv")
    visual_output_path = os.path.join(results_dir, "defect_preservation_visual.png")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Hardware & GPU inspection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    # Load 640 validation images using deterministic split (80/20, seed=42)
    _, val_dataset = create_train_val_datasets(
        root_dir=os.path.join(project_root, "dataset", "KLA", "train", "train"),
        val_ratio=0.20,
        seed=42,
        normalize_gt=True
    )
    num_val_images = len(val_dataset)

    # Instantiate model and load checkpoint
    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Metric accumulators for Bicubic and PixelSilicon
    bicubic_metrics = {"corr": [], "precision": [], "recall": [], "f1": [], "ratio": []}
    pixelsilicon_metrics = {"corr": [], "precision": [], "recall": [], "f1": [], "ratio": []}

    # Selected 5 representative validation sample indices for visualization figure
    vis_indices = [0, 50, 100, 200, 300]
    vis_samples = []  # List of dicts with images & gradient maps

    print("=================================================================", flush=True)
    print(f"Beginning Structural Detail Preservation Evaluation on {num_val_images} Validation Samples...", flush=True)
    print("=================================================================", flush=True)

    start_time = time.time()

    with torch.inference_mode():
        for idx in range(num_val_images):
            noisy_lr_tensor, gt_tensor = val_dataset[idx]

            # Convert input to PyTorch tensor [1, 1, 128, 128]
            noisy_lr_t = noisy_lr_tensor.unsqueeze(0).to(device)

            # Generate PixelSilicon prediction [1, 1, 256, 256]
            pred_t = model(noisy_lr_t)

            # Extract 2D NumPy arrays clamped to [0.0, 1.0]
            noisy_lr_np = noisy_lr_tensor.squeeze().cpu().numpy()  # (128, 128)
            pred_np = torch.clamp(pred_t, 0.0, 1.0).squeeze().cpu().numpy()  # (256, 256)
            gt_np = torch.clamp(gt_tensor, 0.0, 1.0).squeeze().cpu().numpy() # (256, 256)

            # Generate Bicubic 256x256 baseline
            bicubic_np = cv2.resize(noisy_lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)
            bicubic_np = np.clip(bicubic_np, 0.0, 1.0)

            # Compute Sobel Gradient Magnitude Maps
            grad_gt = compute_sobel_gradient(gt_np)
            grad_bicubic = compute_sobel_gradient(bicubic_np)
            grad_pred = compute_sobel_gradient(pred_np)

            # Calculate edge & structural metrics vs GT
            b_corr, b_prec, b_rec, b_f1, b_ratio = compute_edge_metrics(grad_bicubic, grad_gt)
            p_corr, p_prec, p_rec, p_f1, p_ratio = compute_edge_metrics(grad_pred, grad_gt)

            # Accumulate metrics
            bicubic_metrics["corr"].append(b_corr)
            bicubic_metrics["precision"].append(b_prec)
            bicubic_metrics["recall"].append(b_rec)
            bicubic_metrics["f1"].append(b_f1)
            bicubic_metrics["ratio"].append(b_ratio)

            pixelsilicon_metrics["corr"].append(p_corr)
            pixelsilicon_metrics["precision"].append(p_prec)
            pixelsilicon_metrics["recall"].append(p_rec)
            pixelsilicon_metrics["f1"].append(p_f1)
            pixelsilicon_metrics["ratio"].append(p_ratio)

            # Record visualization sample if in vis_indices list
            if idx in vis_indices:
                sample_idx = val_dataset.indices[idx]
                filename = val_dataset.dataset.filenames[sample_idx]
                vis_samples.append({
                    "filename": filename,
                    "bicubic": bicubic_np,
                    "grad_bicubic": grad_bicubic,
                    "pixelsilicon": pred_np,
                    "grad_pixelsilicon": grad_pred,
                    "gt": gt_np,
                    "grad_gt": grad_gt
                })

            if (idx + 1) % 100 == 0 or (idx + 1) == num_val_images:
                print(f"Evaluated {idx + 1:03d} / {num_val_images} validation samples...", flush=True)

    if cuda_available:
        torch.cuda.synchronize()

    total_eval_time = time.time() - start_time

    # Compute dataset-wide average metrics
    b_mean_corr = float(np.mean(bicubic_metrics["corr"]))
    b_mean_prec = float(np.mean(bicubic_metrics["precision"]))
    b_mean_rec = float(np.mean(bicubic_metrics["recall"]))
    b_mean_f1 = float(np.mean(bicubic_metrics["f1"]))
    b_mean_ratio = float(np.mean(bicubic_metrics["ratio"]))

    p_mean_corr = float(np.mean(pixelsilicon_metrics["corr"]))
    p_mean_prec = float(np.mean(pixelsilicon_metrics["precision"]))
    p_mean_rec = float(np.mean(pixelsilicon_metrics["recall"]))
    p_mean_f1 = float(np.mean(pixelsilicon_metrics["f1"]))
    p_mean_ratio = float(np.mean(pixelsilicon_metrics["ratio"]))

    # Edge Preservation Index (EPI = PixelSilicon Edge F1 / Bicubic Edge F1)
    epi = p_mean_f1 / b_mean_f1 if b_mean_f1 > 1e-8 else 0.0

    # Format output summary text
    summary_text = f"""============================================================
DEFECT / STRUCTURAL PRESERVATION EVALUATION
============================================================
Validation Images:                     {num_val_images}
Hardware / GPU Name:                   {gpu_name}
Total Evaluation Time:                 {total_eval_time:.4f} seconds
------------------------------------------------------------
Bicubic:
Gradient Correlation:                  {b_mean_corr:.4f}
Edge Precision:                        {b_mean_prec:.4f}
Edge Recall:                           {b_mean_rec:.4f}
Edge F1:                               {b_mean_f1:.4f}
Gradient Magnitude Ratio:              {b_mean_ratio:.4f}

PixelSilicon:
Gradient Correlation:                  {p_mean_corr:.4f}
Edge Precision:                        {p_mean_prec:.4f}
Edge Recall:                           {p_mean_rec:.4f}
Edge F1:                               {p_mean_f1:.4f}
Gradient Magnitude Ratio:              {p_mean_ratio:.4f}

Edge Preservation Index:
EPI = {epi:.4f}
============================================================
"""

    print("\n" + summary_text, flush=True)

    # Save metrics text report
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"Metrics report saved to: {txt_output_path}", flush=True)

    # Save metrics CSV file
    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "gradient_correlation", "edge_precision", "edge_recall", "edge_f1", "gradient_magnitude_ratio"])
        writer.writerow(["Bicubic", f"{b_mean_corr:.4f}", f"{b_mean_prec:.4f}", f"{b_mean_rec:.4f}", f"{b_mean_f1:.4f}", f"{b_mean_ratio:.4f}"])
        writer.writerow(["PixelSilicon", f"{p_mean_corr:.4f}", f"{p_mean_prec:.4f}", f"{p_mean_rec:.4f}", f"{p_mean_f1:.4f}", f"{p_mean_ratio:.4f}"])

    print(f"Metrics CSV saved to:    {csv_output_path}", flush=True)

    # Generate visual figure for 5 representative validation images (5 rows x 6 columns)
    num_vis = len(vis_samples)
    fig, axes = plt.subplots(num_vis, 6, figsize=(18, 3.2 * num_vis))

    for r_idx, sample in enumerate(vis_samples):
        # Col 1: Bicubic Image
        axes[r_idx, 0].imshow(sample["bicubic"], cmap="gray")
        axes[r_idx, 0].set_title(f"Sample {sample['filename']}\n1. Bicubic Image", fontsize=9, fontweight="bold")
        axes[r_idx, 0].axis("off")

        # Col 2: Bicubic Gradient Map
        axes[r_idx, 1].imshow(sample["grad_bicubic"], cmap="inferno")
        axes[r_idx, 1].set_title("2. Bicubic Grad Map", fontsize=9, fontweight="bold")
        axes[r_idx, 1].axis("off")

        # Col 3: PixelSilicon Image
        axes[r_idx, 2].imshow(sample["pixelsilicon"], cmap="gray")
        axes[r_idx, 2].set_title("3. PixelSilicon Image", fontsize=9, fontweight="bold")
        axes[r_idx, 2].axis("off")

        # Col 4: PixelSilicon Gradient Map
        axes[r_idx, 3].imshow(sample["grad_pixelsilicon"], cmap="inferno")
        axes[r_idx, 3].set_title("4. PixelSilicon Grad Map", fontsize=9, fontweight="bold")
        axes[r_idx, 3].axis("off")

        # Col 5: GT Image
        axes[r_idx, 4].imshow(sample["gt"], cmap="gray")
        axes[r_idx, 4].set_title("5. Ground Truth (GT)", fontsize=9, fontweight="bold")
        axes[r_idx, 4].axis("off")

        # Col 6: GT Gradient Map
        axes[r_idx, 5].imshow(sample["grad_gt"], cmap="inferno")
        axes[r_idx, 5].set_title("6. GT Grad Map", fontsize=9, fontweight="bold")
        axes[r_idx, 5].axis("off")

    plt.suptitle("Structural Detail & Edge Preservation Evaluation (Sobel Gradient Analysis)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(visual_output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visual preservation figure saved to: {visual_output_path}", flush=True)


if __name__ == "__main__":
    evaluate_defect_preservation()
