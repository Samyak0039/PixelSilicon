import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
    # File paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    noisy_lr_path = os.path.join(base_dir, "dataset", "KLA", "train", "train", "NoisyLR", "000000.npy")
    gt_path = os.path.join(base_dir, "dataset", "KLA", "train", "train", "GT", "000000.npy")
    output_dir = os.path.join(base_dir, "results")
    output_path = os.path.join(output_dir, "kla_sample_000000.png")

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset files
    print(f"Loading NoisyLR image from: {noisy_lr_path}")
    noisy_lr = np.load(noisy_lr_path)
    print(f"Loading GT image from: {gt_path}")
    gt = np.load(gt_path)

    # Print image statistics
    print("\n" + "=" * 50)
    print("           KLA DATASET SAMPLE 000000 STATS          ")
    print("=" * 50)

    print("\n--- NoisyLR Image ---")
    print(f"Shape:               {noisy_lr.shape}")
    print(f"Data type (dtype):   {noisy_lr.dtype}")
    print(f"Min value:           {noisy_lr.min()}")
    print(f"Max value:           {noisy_lr.max()}")
    print(f"Mean value:          {noisy_lr.mean():.6f}")
    print(f"Standard deviation:  {noisy_lr.std():.6f}")

    print("\n--- Ground Truth (GT) Image ---")
    print(f"Shape:               {gt.shape}")
    print(f"Data type (dtype):   {gt.dtype}")
    print(f"Min value:           {gt.min()}")
    print(f"Max value:           {gt.max()}")
    print(f"Mean value:          {gt.mean():.6f}")
    print(f"Standard deviation:  {gt.std():.6f}")
    print("=" * 50 + "\n")

    # Ensure 2D spatial dimensions for visualization
    def prepare_array(arr):
        vis_arr = np.squeeze(arr)
        if vis_arr.ndim == 3 and vis_arr.shape[0] in [1, 3]:
            vis_arr = np.transpose(vis_arr, (1, 2, 0))
        return vis_arr

    noisy_lr_2d = prepare_array(noisy_lr)
    gt_2d = prepare_array(gt)

    # Bicubic upscale NoisyLR from 128x128 to 256x256 using OpenCV
    noisy_lr_upscaled = cv2.resize(noisy_lr_2d, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Plot 3 panels in grayscale
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 1: Original NoisyLR (128x128)
    im0 = axes[0].imshow(noisy_lr_2d, cmap='gray')
    axes[0].set_title(f"1. Original NoisyLR\n({noisy_lr_2d.shape[1]}x{noisy_lr_2d.shape[0]})", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: GT (256x256)
    im1 = axes[1].imshow(gt_2d, cmap='gray')
    axes[1].set_title(f"2. Ground Truth (GT)\n({gt_2d.shape[1]}x{gt_2d.shape[0]})", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: NoisyLR Bicubic Upscaled (256x256)
    im2 = axes[2].imshow(noisy_lr_upscaled, cmap='gray')
    axes[2].set_title(f"3. NoisyLR Bicubic Upscaled\n({noisy_lr_upscaled.shape[1]}x{noisy_lr_upscaled.shape[0]})", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("KLA Dataset Inspection - Sample 000000", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save output visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
