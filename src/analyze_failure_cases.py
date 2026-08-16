import os
import sys
import csv
import time
from typing import Tuple, List, Dict

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from src.dataset import create_train_val_datasets
from src.model import PixelSiliconNet


def find_high_gradient_crop(gt_img: np.ndarray, crop_sz: int = 64) -> Tuple[int, int]:
    """
    Automatically selects the (y_min, x_min) crop window of size crop_sz x crop_sz
    in gt_img that maximizes Sobel gradient magnitude sum (high structural detail region).
    """
    h, w = gt_img.shape
    grad_x = cv2.Sobel(gt_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    best_score = -1.0
    best_crop = (h // 2 - crop_sz // 2, w // 2 - crop_sz // 2)

    for y in range(0, h - crop_sz + 1, 8):
        for x in range(0, w - crop_sz + 1, 8):
            crop = grad_mag[y:y + crop_sz, x:x + crop_sz]
            score = float(np.sum(crop))
            if score > best_score:
                best_score = score
                best_crop = (y, x)

    return best_crop


def analyze_failure_cases():
    """
    Performs per-sample metric calculation across all 640 KLA validation images.
    Objectively identifies the Best Case, Worst PSNR Improvement Case, and Worst SSIM Improvement Case.
    Saves outputs to CSV, TXT summary, comparison figure, and summary table figure.
    """
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "pixelsilicon_best_stable.pth")
    csv_out_path = os.path.join(results_dir, "failure_case_analysis.csv")
    txt_out_path = os.path.join(results_dir, "failure_case_analysis.txt")
    comparison_fig_path = os.path.join(results_dir, "failure_case_comparison.png")
    summary_fig_path = os.path.join(results_dir, "failure_case_summary.png")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # Hardware & GPU inspection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    # 1 & 2. Load exact 640 validation images using deterministic split (80/20, seed=42)
    _, val_dataset = create_train_val_datasets(
        root_dir=os.path.join(project_root, "dataset", "KLA", "train", "train"),
        val_ratio=0.20,
        seed=42,
        normalize_gt=True
    )
    num_val_images = len(val_dataset)

    # Load PixelSiliconNet architecture and checkpoint weights
    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    per_sample_results = []

    print("=================================================================", flush=True)
    print(f"Beginning Objective Per-Sample Metrics Analysis on {num_val_images} Validation Images...", flush=True)
    print("=================================================================", flush=True)

    start_time = time.time()

    # 3 & 4. Per-sample inference and metric calculation under torch.inference_mode()
    with torch.inference_mode():
        for idx in range(num_val_images):
            noisy_lr_tensor, gt_tensor = val_dataset[idx]

            sample_idx = val_dataset.indices[idx]
            filename = val_dataset.dataset.filenames[sample_idx]
            sample_id = os.path.splitext(filename)[0]

            noisy_lr_t = noisy_lr_tensor.unsqueeze(0).to(device)
            pred_t = model(noisy_lr_t)

            noisy_lr_np = noisy_lr_tensor.squeeze().cpu().numpy()
            pred_np = torch.clamp(pred_t, 0.0, 1.0).squeeze().cpu().numpy()
            gt_np = torch.clamp(gt_tensor, 0.0, 1.0).squeeze().cpu().numpy()

            # Bicubic 256x256 baseline
            bicubic_np = cv2.resize(noisy_lr_np, (256, 256), interpolation=cv2.INTER_CUBIC)
            bicubic_clamped = np.clip(bicubic_np, 0.0, 1.0)

            # Per-sample PSNR & SSIM metrics
            ps_psnr = float(compute_psnr(gt_np, pred_np, data_range=1.0))
            bi_psnr = float(compute_psnr(gt_np, bicubic_clamped, data_range=1.0))
            psnr_diff = ps_psnr - bi_psnr

            ps_ssim = float(compute_ssim(gt_np, pred_np, data_range=1.0))
            bi_ssim = float(compute_ssim(gt_np, bicubic_clamped, data_range=1.0))
            ssim_diff = ps_ssim - bi_ssim

            per_sample_results.append({
                "val_idx": idx,
                "sample_id": sample_id,
                "filename": filename,
                "pixelsilicon_psnr": ps_psnr,
                "bicubic_psnr": bi_psnr,
                "psnr_improvement": psnr_diff,
                "pixelsilicon_ssim": ps_ssim,
                "bicubic_ssim": bi_ssim,
                "ssim_improvement": ssim_diff,
                "noisy_lr_np": noisy_lr_np,
                "bicubic_np": bicubic_clamped,
                "pred_np": pred_np,
                "gt_np": gt_np
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == num_val_images:
                print(f"Analyzed {idx + 1:03d} / {num_val_images} samples...", flush=True)

    if cuda_available:
        torch.cuda.synchronize()

    eval_duration = time.time() - start_time

    # 5. Save all 640 per-sample results to results/failure_case_analysis.csv
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id", "pixelsilicon_psnr", "bicubic_psnr", "psnr_improvement",
            "pixelsilicon_ssim", "bicubic_ssim", "ssim_improvement"
        ])
        writer.writeheader()
        for r in per_sample_results:
            writer.writerow({
                "sample_id": r["sample_id"],
                "pixelsilicon_psnr": f"{r['pixelsilicon_psnr']:.4f}",
                "bicubic_psnr": f"{r['bicubic_psnr']:.4f}",
                "psnr_improvement": f"{r['psnr_improvement']:.4f}",
                "pixelsilicon_ssim": f"{r['pixelsilicon_ssim']:.4f}",
                "bicubic_ssim": f"{r['bicubic_ssim']:.4f}",
                "ssim_improvement": f"{r['ssim_improvement']:.4f}"
            })

    print(f"\nSaved all 640 per-sample analysis results to: {csv_out_path}", flush=True)

    # 6 & 7. Identify Best, Worst PSNR, and Worst SSIM Cases & Check for Actual Failure Cases
    best_case = max(per_sample_results, key=lambda x: x["psnr_improvement"])
    worst_psnr_case = min(per_sample_results, key=lambda x: x["psnr_improvement"])
    worst_ssim_case = min(per_sample_results, key=lambda x: x["ssim_improvement"])

    actual_failures = [r for r in per_sample_results if r["psnr_improvement"] < 0.0]
    num_actual_failures = len(actual_failures)

    limitation_label = "Actual Bicubic-Outperforming Failure Case" if num_actual_failures > 0 else "Weakest Relative-Improvement / Limitation Case"

    # 8 & 12. Print and Save machine-readable text report to results/failure_case_analysis.txt
    txt_lines = [
        "============================================================",
        "FAILURE / LIMITATION CASE ANALYSIS REPORT",
        "============================================================",
        f"Validation samples:                    {num_val_images}",
        f"Execution Device:                      {gpu_name}",
        f"Analysis Duration:                     {eval_duration:.2f} seconds",
        "------------------------------------------------------------",
        "BEST CASE (Highest PSNR Improvement over Bicubic):",
        f"Sample ID:                             {best_case['sample_id']}",
        f"PixelSilicon PSNR:                      {best_case['pixelsilicon_psnr']:.4f} dB",
        f"Bicubic PSNR:                          {best_case['bicubic_psnr']:.4f} dB",
        f"Improvement:                           +{best_case['psnr_improvement']:.4f} dB",
        f"PixelSilicon SSIM:                      {best_case['pixelsilicon_ssim']:.4f}",
        f"Bicubic SSIM:                          {best_case['bicubic_ssim']:.4f}",
        f"SSIM Improvement:                      +{best_case['ssim_improvement']:.4f}",
        "------------------------------------------------------------",
        f"WORST PSNR IMPROVEMENT ({limitation_label}):",
        f"Sample ID:                             {worst_psnr_case['sample_id']}",
        f"PixelSilicon PSNR:                      {worst_psnr_case['pixelsilicon_psnr']:.4f} dB",
        f"Bicubic PSNR:                          {worst_psnr_case['bicubic_psnr']:.4f} dB",
        f"Improvement:                           {worst_psnr_case['psnr_improvement']:+.4f} dB",
        f"PixelSilicon SSIM:                      {worst_psnr_case['pixelsilicon_ssim']:.4f}",
        f"Bicubic SSIM:                          {worst_psnr_case['bicubic_ssim']:.4f}",
        f"SSIM Improvement:                      {worst_psnr_case['ssim_improvement']:+.4f}",
        "------------------------------------------------------------",
        "WORST SSIM IMPROVEMENT:",
        f"Sample ID:                             {worst_ssim_case['sample_id']}",
        f"PixelSilicon PSNR:                      {worst_ssim_case['pixelsilicon_psnr']:.4f} dB",
        f"Bicubic PSNR:                          {worst_ssim_case['bicubic_psnr']:.4f} dB",
        f"Improvement:                           {worst_ssim_case['psnr_improvement']:+.4f} dB",
        f"PixelSilicon SSIM:                      {worst_ssim_case['pixelsilicon_ssim']:.4f}",
        f"Bicubic SSIM:                          {worst_ssim_case['bicubic_ssim']:.4f}",
        f"SSIM Improvement:                      {worst_ssim_case['ssim_improvement']:+.4f}",
        "------------------------------------------------------------",
        "ACTUAL BICUBIC-OUTPERFORMED CASES (PixelSilicon PSNR < Bicubic PSNR):",
        f"Count:                                 {num_actual_failures}",
        "============================================================"
    ]
    txt_report = "\n".join(txt_lines)

    print("\n" + txt_report, flush=True)

    with open(txt_out_path, "w", encoding="utf-8") as f:
        f.write(txt_report + "\n")
    print(f"Machine-readable text summary saved to: {txt_out_path}", flush=True)

    # 9, 10, 11. Generate failure_case_comparison.png (2 rows x 4 columns with automatic high-gradient crop)
    cases_to_plot = [
        ("Row 1: BEST CASE", best_case),
        (f"Row 2: {limitation_label.upper()}", worst_psnr_case)
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 17))

    for r_idx, (row_label, c_data) in enumerate(cases_to_plot):
        y_min, x_min = find_high_gradient_crop(c_data["gt_np"], crop_sz=64)
        crop_sz = 64

        noisy_raw = cv2.resize(c_data["noisy_lr_np"], (256, 256), interpolation=cv2.INTER_NEAREST)
        noisy_raw_clamped = np.clip(noisy_raw, 0.0, 1.0)

        imgs = [
            (noisy_raw_clamped, "NoisyLR (256x256)", "Input: Raw 128x128\nSensor Noise"),
            (c_data["bicubic_np"], "Bicubic Baseline", f"PSNR: {c_data['bicubic_psnr']:.2f} dB\nSSIM: {c_data['bicubic_ssim']:.4f}"),
            (c_data["pred_np"], "PixelSilicon Restoration", f"PSNR: {c_data['pixelsilicon_psnr']:.2f} dB ({c_data['psnr_improvement']:+.2f} dB)\nSSIM: {c_data['pixelsilicon_ssim']:.4f}"),
            (c_data["gt_np"], "Ground Truth (GT)", "Target Reference\nNormalized [0, 1]")
        ]

        r_full = r_idx * 2
        r_crop = r_idx * 2 + 1

        for c_idx, (img, col_title, caption) in enumerate(imgs):
            # Full Image Panel
            ax_f = axes[r_full, c_idx]
            ax_f.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            rect = patches.Rectangle((x_min, y_min), crop_sz, crop_sz, linewidth=2, edgecolor="red", facecolor="none")
            ax_f.add_patch(rect)
            if r_idx == 0:
                ax_f.set_title(col_title, fontsize=12, fontweight="bold", pad=8)
            if c_idx == 0:
                ax_f.set_ylabel(f"{row_label}\nSample {c_data['sample_id']}\nFull Image", fontsize=10, fontweight="bold")
            ax_f.set_xlabel(caption, fontsize=9, color="#1E293B", fontweight="semibold")
            ax_f.set_xticks([])
            ax_f.set_yticks([])

            # Zoomed Detail Crop Panel
            crop_img = img[y_min:y_min + crop_sz, x_min:x_min + crop_sz]
            ax_c = axes[r_crop, c_idx]
            ax_c.imshow(crop_img, cmap="gray", vmin=0.0, vmax=1.0)
            for spine in ax_c.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
            if c_idx == 0:
                ax_c.set_ylabel("Zoomed Detail Crop\n(Coords: Y=%d, X=%d)" % (y_min, x_min), fontsize=9, fontweight="bold")
            ax_c.set_xticks([])
            ax_c.set_yticks([])

    plt.suptitle("KLA Validation Analysis: Best Case vs Limitation Case Comparison", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(comparison_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Visual comparison plot saved to: {comparison_fig_path}", flush=True)

    # 13. Generate failure_case_summary.png (Concise Matplotlib summary table figure)
    fig_sum, ax_sum = plt.subplots(figsize=(11, 4.5))
    ax_sum.axis("off")

    table_data = [
        ["Category", "Sample ID", "PixelSilicon PSNR", "Bicubic PSNR", "PSNR Delta", "PixelSilicon SSIM", "Bicubic SSIM"],
        ["Best Case", best_case["sample_id"], f"{best_case['pixelsilicon_psnr']:.4f} dB", f"{best_case['bicubic_psnr']:.4f} dB", f"+{best_case['psnr_improvement']:.4f} dB", f"{best_case['pixelsilicon_ssim']:.4f}", f"{best_case['bicubic_ssim']:.4f}"],
        ["Worst PSNR Delta", worst_psnr_case["sample_id"], f"{worst_psnr_case['pixelsilicon_psnr']:.4f} dB", f"{worst_psnr_case['bicubic_psnr']:.4f} dB", f"{worst_psnr_case['psnr_improvement']:+.4f} dB", f"{worst_psnr_case['pixelsilicon_ssim']:.4f}", f"{worst_psnr_case['bicubic_ssim']:.4f}"],
        ["Worst SSIM Delta", worst_ssim_case["sample_id"], f"{worst_ssim_case['pixelsilicon_psnr']:.4f} dB", f"{worst_ssim_case['bicubic_psnr']:.4f} dB", f"{worst_ssim_case['psnr_improvement']:+.4f} dB", f"{worst_ssim_case['pixelsilicon_ssim']:.4f}", f"{worst_ssim_case['bicubic_ssim']:.4f}"],
        ["Actual Failure Count", f"{num_actual_failures} / 640", "-", "-", "-", "-", "-"]
    ]

    table = ax_sum.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for col in range(len(table_data[0])):
        table[(0, col)].set_facecolor("#1E293B")
        table[(0, col)].get_text().set_color("white")
        table[(0, col)].get_text().set_weight("bold")

    plt.title("PixelSilicon KLA Validation Extreme Cases & Failure Summary", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(summary_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"Summary table figure saved to:  {summary_fig_path}", flush=True)


if __name__ == "__main__":
    analyze_failure_cases()
