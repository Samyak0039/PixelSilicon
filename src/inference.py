import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch

# Resolve project root directory robustly so script can be launched from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import PixelSiliconNet


def parse_args():
    """
    Parses command-line arguments for standalone KLA inference.
    Required: --input_dir, --output_dir
    Optional: --checkpoint
    """
    parser = argparse.ArgumentParser(
        description="PixelSilicon — Standalone KLA Super-Resolution & Restoration Inference Pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing input degraded .npy files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where restored .png images will be saved."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(project_root, "models", "pixelsilicon_best_stable.pth"),
        help="Path to trained PyTorch model checkpoint (.pth file)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    checkpoint_path = os.path.abspath(args.checkpoint)

    # 14. Validation checks & error handling
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Error: Input directory does not exist: {input_dir}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Error: Checkpoint file does not exist: {checkpoint_path}")

    # 10. Create output directory automatically if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # 5. Automatically detect CUDA / CPU
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    # 4. Instantiate exact PixelSiliconNet architecture
    model = PixelSiliconNet(
        in_channels=1,
        out_channels=1,
        num_features=64,
        num_blocks=8,
        upscale_factor=2
    )

    # 3. Load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 6. Read ALL .npy files from --input_dir (ignore hidden files, sort deterministically)
    test_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".npy") and not f.startswith(".")
    ])
    num_files = len(test_files)

    if num_files == 0:
        raise ValueError(f"Error: No .npy files found in input directory: {input_dir}")

    # 12. Print clear startup configuration
    print("=================================================================", flush=True)
    print("         PIXELSILICON KLA STANDALONE INFERENCE PIPELINE          ", flush=True)
    print("=================================================================", flush=True)
    print(f"Input Directory:           {input_dir}", flush=True)
    print(f"Output Directory:          {output_dir}", flush=True)
    print(f"Model Checkpoint:          {checkpoint_path}", flush=True)
    print(f"Execution Device:          {device_name}", flush=True)
    print(f"Model Parameters:          {total_params:,} ({total_params / 1e6:.4f} M)", flush=True)
    print(f"Input Files Count:         {num_files}", flush=True)
    print("=================================================================\n", flush=True)

    # Warmup GPU memory allocator
    dummy_input = torch.zeros(1, 1, 128, 128, device=device)
    with torch.inference_mode():
        _ = model(dummy_input)
    if cuda_available:
        torch.cuda.synchronize()

    # 7. Run inference under torch.inference_mode()
    print("Processing input images...", flush=True)
    start_time = time.time()

    with torch.inference_mode():
        for filename in test_files:
            file_path = os.path.join(input_dir, filename)

            # Load 2D grayscale array as float32
            arr = np.load(file_path).astype(np.float32)

            if arr.ndim != 2:
                raise ValueError(
                    f"Error: File '{filename}' has shape {arr.shape} with {arr.ndim} dimensions. "
                    f"Expected a 2D grayscale array [H, W]."
                )

            # Convert array to PyTorch tensor with shape [1, 1, H, W]
            input_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            # Model forward pass
            output_tensor = model(input_tensor)

            # 8. Convert model output float32 [0, 1] to uint8 [0, 255]
            pred_np = output_tensor.squeeze().cpu().numpy()
            uint8_img = np.clip(pred_np * 255.0, 0.0, 255.0).astype(np.uint8)

            # 9. Save output PNG image matching filename base
            sample_id = os.path.splitext(filename)[0]
            out_filepath = os.path.join(output_dir, f"{sample_id}.png")
            cv2.imwrite(out_filepath, uint8_img)

    if cuda_available:
        torch.cuda.synchronize()

    # 12. Calculate final statistics
    total_time = time.time() - start_time
    avg_time_ms = (total_time / num_files) * 1000.0
    fps = num_files / total_time

    print("\n=================================================================", flush=True)
    print("                     INFERENCE COMPLETE                          ", flush=True)
    print("=================================================================", flush=True)
    print(f"Processed Images:          {num_files}", flush=True)
    print(f"Total Inference Time:       {total_time:.4f} seconds", flush=True)
    print(f"Average Time per Image:     {avg_time_ms:.2f} ms", flush=True)
    print(f"Throughput (FPS):           {fps:.2f} images/sec", flush=True)
    print(f"Restored Images Saved to:   {output_dir}", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
