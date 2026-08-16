import os
import sys
import numpy as np

# Ensure project root directory is on Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def inspect_test_dataset():
    test_dir = os.path.join(project_root, "dataset", "KLA", "Test_NoisyLR", "NoisyLR")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file_path = os.path.join(results_dir, "test_dataset_info.txt")

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    # 1. Count total number of .npy files
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.npy') and not f.startswith('.')])
    total_files = len(files)

    if total_files == 0:
        raise ValueError(f"No .npy files found in: {test_dir}")

    # 2. Load the first test .npy file
    first_filename = files[0]
    first_filepath = os.path.join(test_dir, first_filename)
    first_array = np.load(first_filepath)

    # Calculate statistics
    arr_shape = first_array.shape
    arr_dtype = str(first_array.dtype)
    arr_min = float(np.min(first_array))
    arr_max = float(np.max(first_array))
    arr_mean = float(np.mean(first_array))
    arr_std = float(np.std(first_array))

    # 4. Confirm input shape compatibility with [1, 128, 128]
    # Standard 2D image shape is expected to be (128, 128)
    is_compatible = (arr_shape == (128, 128))
    compat_msg = (
        "CONFIRMED: Array shape (128, 128) is fully compatible with PixelSiliconNet's "
        "expected input tensor shape of [1, 1, 128, 128] (batch_size=1, channels=1, H=128, W=128)."
        if is_compatible
        else f"WARNING: Unexpected array shape {arr_shape}."
    )

    info_lines = [
        "=================================================================",
        "                 KLA TEST DATASET INSPECTION                     ",
        "=================================================================",
        f"Test Directory:             {test_dir}",
        f"Total .npy Files Count:     {total_files}",
        "-----------------------------------------------------------------",
        "FIRST TEST SAMPLE ANALYSIS:",
        f"Filename:                   {first_filename}",
        f"Shape:                      {arr_shape}",
        f"Dtype:                      {arr_dtype}",
        f"Minimum Value:              {arr_min:.6f}",
        f"Maximum Value:              {arr_max:.6f}",
        f"Mean Value:                 {arr_mean:.6f}",
        f"Standard Deviation:         {arr_std:.6f}",
        "-----------------------------------------------------------------",
        "INPUT SHAPE COMPATIBILITY CHECK:",
        f"{compat_msg}",
        "================================================================="
    ]

    info_text = "\n".join(info_lines)

    # Print inspection output
    print(info_text, flush=True)

    # Save inspection info to results/test_dataset_info.txt
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(info_text + "\n")

    print(f"\nInspection details saved successfully to: {out_file_path}", flush=True)


if __name__ == "__main__":
    inspect_test_dataset()
