# PixelSilicon

## AI-Based Restoration and 2× Super-Resolution of Degraded Semiconductor Inspection Images

PixelSilicon is a lightweight deep-learning-based image restoration and super-resolution system designed for noisy, low-resolution semiconductor inspection images.

The system learns a supervised mapping from degraded low-resolution inspection images to high-resolution ground-truth references, with an emphasis on recovering structural details and preserving important edges.

---

## 1. Problem Statement

Semiconductor inspection images can be affected by:

- Speckle noise
- Additive Gaussian noise
- Low spatial resolution
- Downsampling
- Blur and interpolation artifacts
- Loss of fine structural details

These degradations can make small semiconductor features and structural abnormalities more difficult to inspect.

Traditional interpolation methods such as bicubic upscaling can increase spatial resolution, but they cannot learn to recover missing high-frequency information.

PixelSilicon addresses this problem using a learned image restoration and 2× super-resolution model.

---

## 2. Objective

The main objectives of PixelSilicon are:

1. Restore degraded semiconductor inspection images.
2. Perform 2× spatial super-resolution.
3. Recover structural and edge information lost during degradation.
4. Improve image quality compared with conventional bicubic interpolation.
5. Evaluate restoration using PSNR, SSIM, and LPIPS.
6. Evaluate structural-detail preservation using gradient and edge-based metrics.
7. Provide a reproducible standalone inference pipeline.
8. Demonstrate practical GPU inference performance.

---

## 3. Final System Pipeline

The final AI inference pipeline is:

```text
KLA NoisyLR Image
      │
      │ 128 × 128
      ▼
PixelSiliconNet
      │
      │ Learned Restoration
      │ + 2× Super-Resolution
      ▼
Restored Image
      │
      │ 256 × 256
      ▼
Image Quality / Structural Evaluation
