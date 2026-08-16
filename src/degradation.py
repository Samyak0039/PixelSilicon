import os
import glob
import numpy as np
import cv2


def apply_degradation(
    image: np.ndarray,
    scale_factor: float = 0.5,
    gaussian_std: float = 12.0,
    speckle_std: float = 0.08,
    seed: int = 42
) -> np.ndarray:
    """
    Applies a synthetic semiconductor inspection degradation pipeline to an input image.

    Degradation Stages:
    -------------------
    1. Downsampling (Low-Resolution Simulation):
       Simulates spatial resolution loss of high-speed inspection optics.
       I_lr = resize(I_orig, (W * scale_factor, H * scale_factor), INTER_AREA)

    2. Bicubic Upscaling (Dimension Restoration):
       Restores image back to original resolution (W, H) using bicubic interpolation.
       I_up = resize(I_lr, (W, H), INTER_CUBIC)
       This introduces characteristic low-pass optical blurring artifacts.

    3. Additive Gaussian Noise (Sensor Readout Noise):
       Simulates electronic readout & thermal amplifier noise.
       n_gauss ~ N(0, gaussian_std^2)
       I_gauss = I_up + n_gauss

    4. Multiplicative Speckle Noise (Coherent Surface Scattering):
       Simulates laser/beam speckle interference on polished silicon surfaces.
       n_speckle ~ N(0, speckle_std^2)
       I_speckle = I_gauss * (1.0 + n_speckle)

    5. Dynamic Range Clipping & Quantization:
       Clips intensity values to [0, 255] and converts back to uint8.

    Parameters:
    -----------
    image : np.ndarray
        Input grayscale or color image (uint8).
    scale_factor : float
        Downsampling scale ratio (0.0 < scale_factor <= 1.0). Default is 0.5 (2x downsampling).
    gaussian_std : float
        Standard deviation of additive Gaussian noise (in pixel intensity units). Default is 12.0.
    speckle_std : float
        Standard deviation of multiplicative speckle noise (relative noise factor). Default is 0.08.
    seed : int
        Random seed for reproducible noise generation. Default is 42.

    Returns:
    --------
    degraded_image : np.ndarray
        Degraded image with identical dimensions and uint8 datatype.
    """
    if image is None:
        raise ValueError("Input image is None.")

    # Set random seed for reproducible noise generation
    np.random.seed(seed)

    orig_height, orig_width = image.shape[:2]

    # Calculate downsampled dimensions
    new_width = max(1, int(orig_width * scale_factor))
    new_height = max(1, int(orig_height * scale_factor))

    # --- Step 1: Downsampling (Low-Resolution Simulation) ---
    # INTER_AREA is ideal for decimation/downsampling as it averages pixels, simulating camera binning.
    downsampled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # --- Step 2: Bicubic Upscaling (Dimension Matching) ---
    # INTER_CUBIC scales back up to (orig_width, orig_height), leaving smooth anti-aliased blurring.
    upscaled = cv2.resize(downsampled, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)

    # Convert to float32 to prevent underflow/overflow during noise arithmetic
    img_float = upscaled.astype(np.float32)

    # --- Step 3: Additive Gaussian Noise (Thermal / Readout Noise) ---
    # Formula: I_gauss = I_up + N(0, gaussian_std^2)
    gaussian_noise = np.random.normal(loc=0.0, scale=gaussian_std, size=img_float.shape).astype(np.float32)
    img_gauss = img_float + gaussian_noise

    # --- Step 4: Multiplicative Speckle Noise (Coherent Interference Noise) ---
    # Formula: I_degraded = I_gauss * (1 + N(0, speckle_std^2))
    speckle_noise = np.random.normal(loc=0.0, scale=speckle_std, size=img_float.shape).astype(np.float32)
    img_speckle = img_gauss * (1.0 + speckle_noise)

    # --- Step 5: Dynamic Range Clipping & Type Conversion ---
    # Ensure intensity range remains within valid [0, 255] 8-bit image bounds
    degraded_uint8 = np.clip(img_speckle, 0, 255).astype(np.uint8)

    return degraded_uint8


def process_dataset(
    input_dir: str = "dataset/original",
    output_dir: str = "dataset/degraded",
    scale_factor: float = 0.5,
    gaussian_std: float = 12.0,
    speckle_std: float = 0.08,
    seed: int = 42
):
    """
    Processes all original images in input_dir, applies the degradation pipeline,
    and saves the output images to output_dir while preserving original files unchanged.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No image files found in '{input_dir}'.")
        return

    print(f"Found {len(image_paths)} image(s) in '{input_dir}'. Processing degradation...")
    print(f"Parameters: scale_factor={scale_factor}, gaussian_std={gaussian_std}, speckle_std={speckle_std}, seed={seed}")

    for idx, path in enumerate(image_paths):
        filename = os.path.basename(path)
        save_path = os.path.join(output_dir, filename)

        # Read image in unchanged mode (supports grayscale and color)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read image '{path}'. Skipping.")
            continue

        # Apply degradation pipeline
        degraded_img = apply_degradation(
            img,
            scale_factor=scale_factor,
            gaussian_std=gaussian_std,
            speckle_std=speckle_std,
            seed=seed + idx  # deterministic offset per image
        )

        # Save degraded image to output directory
        cv2.imwrite(save_path, degraded_img)
        print(f"Saved degraded image: {save_path} [Shape: {degraded_img.shape}]")

    print(f"Degradation processing complete. Degraded dataset saved in '{output_dir}'.")


def main():
    """Main execution function."""
    process_dataset()


if __name__ == "__main__":
    main()
