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

from src.model import PixelSiliconNet


def run_test_inference():
    """
    Final inference pipeline for PixelSiliconNet on the KLA test dataset.
    Processes all 400 test images, converts output tensors to uint8 PNG images,
    computes timing throughput metrics, and generates a side-by-side visual comparison.
    """
    # 1. Setup paths
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")
    test_inputs_dir = os.path.join(project_root, "dataset", "KLA", "Test_NoisyLR", "NoisyLR")
    test_outputs_dir = os.path.join(results_dir, "test_outputs")

    os.makedirs(test_outputs_dir, exist_ok=True)

    checkpoint_path = os.path.join(models_dir, "pixelsilicon_best_stable.pth")
    stats_file_path = os.path.join(results_dir, "test_inference_stats.txt")
    comparison_plot_path = os.path.join(results_dir, "test_visual_comparison.png")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    # 3. Hardware inspection & GPU detection
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    # 1 & 2. Instantiate PixelSiliconNet architecture and load trained weights
    model = PixelSiliconNet(in_channels=1, out_channels=1, num_features=64, num_blocks=8, upscale_factor=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Find and sort all 400 test .npy files
    test_files = sorted([f for f in os.listdir(test_inputs_dir) if f.endswith(".npy") and not f.startswith(".")])
    num_test_images = len(test_files)

    if num_test_images == 0:
        raise ValueError(f"No .npy test files found in: {test_inputs_dir}")

    # Specific sample indices for visual comparison panel (13 & 14)
    comparison_indices = ["000000", "000050", "000100", "000200", "000300"]
    comparison_data = {}  # Maps sample_id -> (bicubic_np, pred_np)

    # Warm up GPU memory allocator for precise timing measurement
    dummy_input = torch.zeros(1, 1, 128, 128, device=device)
    with torch.inference_mode():
        _ = model(dummy_input)
    if cuda_available:
        torch.cuda.synchronize()

    # 4 & 19. Perform inference on all test images using inference_mode
    print("=================================================================", flush=True)
    print(f"Beginning Final Inference on {num_test_images} KLA Test Images...", flush=True)
    print("=================================================================", flush=True)

    start_time = time.time()

    with torch.inference_mode():
        for filename in test_files:
            sample_id = filename.replace(".npy", "")
            filepath = os.path.join(test_inputs_dir, filename)

            # 6. Load 128x128 float32 input array
            noisy_np = np.load(filepath).astype(np.float32)

            # Convert array to PyTorch float32 tensor [1, 1, 128, 128]
            input_tensor = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).to(device)

            # 7. Model inference producing [1, 1, 256, 256] grayscale output
            output_tensor = model(input_tensor)

            # Extract 2D prediction array [256, 256] in [0, 1] range
            pred_np = output_tensor.squeeze().cpu().numpy()

            # 8. Convert model output from float32 [0,1] to uint8 [0,255]
            uint8_img = np.clip(pred_np * 255.0, 0.0, 255.0).astype(np.uint8)

            # 9. Save output to results/test_outputs/000xxx.png
            save_path = os.path.join(test_outputs_dir, f"{sample_id}.png")
            cv2.imwrite(save_path, uint8_img)

            # Store selected samples for visual comparison plot
            if sample_id in comparison_indices:
                bicubic_np = cv2.resize(noisy_np, (256, 256), interpolation=cv2.INTER_CUBIC)
                bicubic_clamped = np.clip(bicubic_np, 0.0, 1.0)
                comparison_data[sample_id] = (bicubic_clamped, pred_np)

    if cuda_available:
        torch.cuda.synchronize()

    # 10. Measure timing statistics
    total_time = time.time() - start_time
    avg_time_per_img_ms = (total_time / num_test_images) * 1000.0
    fps = num_test_images / total_time

    # 11. Format summary output text
    summary_lines = [
        "=================================================================",
        "           PIXELSILICONNET TEST INFERENCE STATISTICS             ",
        "=================================================================",
        f"Hardware / GPU Name:        {gpu_name}",
        f"Model Checkpoint Loaded:    {checkpoint_path}",
        f"Model Parameter Count:      {total_params:,} ({total_params / 1e6:.4f} M)",
        f"Number of Test Images:      {num_test_images}",
        f"Model Input Tensor Shape:   [1, 1, 128, 128]",
        f"Model Output Image Shape:   256x256 Grayscale",
        "-----------------------------------------------------------------",
        "TIMING STATISTICS:",
        f"Total Inference Time:       {total_time:.4f} seconds",
        f"Average Time per Image:     {avg_time_per_img_ms:.2f} ms",
        f"Throughput (FPS):           {fps:.2f} images/sec",
        "================================================================="
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text, flush=True)

    # 12. Save timing information to results/test_inference_stats.txt
    with open(stats_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nTiming statistics saved to: {stats_file_path}", flush=True)

    # 13 & 14. Create visual comparison plot (Bicubic vs PixelSilicon) for 000000, 000050, 000100, 000200, 000300
    num_samples = len(comparison_indices)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3.8 * num_samples))

    for idx, sample_id in enumerate(comparison_indices):
        if sample_id in comparison_data:
            bicubic_img, pred_img = comparison_data[sample_id]

            # Column 1: Bicubic Upscaled
            ax_bicubic = axes[idx, 0]
            im0 = ax_bicubic.imshow(bicubic_img, cmap="gray")
            ax_bicubic.set_title(f"Sample {sample_id} - Bicubic (256x256)", fontsize=11, fontweight="bold")
            ax_bicubic.axis("off")
            fig.colorbar(im0, ax=ax_bicubic, fraction=0.046, pad=0.04)

            # Column 2: PixelSilicon Restored
            ax_pred = axes[idx, 1]
            im1 = ax_pred.imshow(pred_img, cmap="gray")
            ax_pred.set_title(f"Sample {sample_id} - PixelSilicon (256x256)", fontsize=11, fontweight="bold")
            ax_pred.axis("off")
            fig.colorbar(im1, ax=ax_pred, fraction=0.046, pad=0.04)

    plt.suptitle("KLA Test Dataset Restoration Comparison: Bicubic vs PixelSilicon", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visual comparison plot saved to: {comparison_plot_path}", flush=True)


if __name__ == "__main__":
    run_test_inference()
