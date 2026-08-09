# PixelSilicon

## AI-Based Restoration and Super-Resolution of Degraded Semiconductor Inspection Images

PixelSilicon is an AI-based image restoration project designed to improve degraded semiconductor inspection images.

### Problem

Semiconductor inspection images may suffer from:

- Speckle noise
- Gaussian noise
- Low resolution
- Blur
- Loss of fine structural details

These degradations can make small semiconductor features and defects difficult to inspect.

### Proposed Solution

Our system uses a deep-learning-based image restoration and super-resolution pipeline.

```text
Degraded Semiconductor Image
            ↓
    Degradation Analysis
            ↓
   AI Image Restoration
            ↓
     Super-Resolution
            ↓
    Enhanced Image
            ↓
 PSNR / SSIM / LPIPS
