import os
import numpy as np
import cv2

def generate_synthetic_semiconductor_image(output_path="dataset/original/test_semiconductor.png"):
    """
    Generates a synthetic semiconductor wafer inspection grayscale image.
    Uses pure algorithmic geometry (NumPy & OpenCV) - NO AI models.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Base Setup & Wafer Background
    width, height = 1024, 1024
    img = np.full((height, width), 30, dtype=np.uint8)  # Outer background (dark chamber/stage)
    
    center = (width // 2, height // 2)
    radius = 470
    
    # Create circular wafer mask
    y_indices, x_indices = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2)
    wafer_mask = dist_from_center <= radius
    
    # Wafer surface base intensity
    img[wafer_mask] = 75
    
    # Add wafer alignment notch at bottom edge
    notch_mask = (dist_from_center <= radius + 5) & (y_indices > center[1] + radius - 15) & (np.abs(x_indices - center[0]) < 25)
    img[notch_mask] = 30
    
    # Add subtle radial illumination / vignetting on wafer surface
    vignette = 1.0 - 0.25 * (dist_from_center / radius)**2
    vignette = np.clip(vignette, 0.7, 1.0)
    wafer_float = img.astype(np.float32)
    wafer_float[wafer_mask] *= vignette[wafer_mask]
    img = np.clip(wafer_float, 0, 255).astype(np.uint8)
    
    # Outer wafer rim / bevel edge
    cv2.circle(img, center, radius, 170, 2)
    cv2.circle(img, center, radius - 2, 110, 1)
    
    # 2. Die Grid & Circuit Patterns
    # Define a 5x5 die array across the wafer
    die_size = 140
    street_width = 16  # Scribe lines / streets between dies
    grid_start_x = center[0] - 2 * die_size - 2 * street_width - die_size // 2
    grid_start_y = center[1] - 2 * die_size - 2 * street_width - die_size // 2
    
    for row in range(5):
        for col in range(5):
            dx = grid_start_x + col * (die_size + street_width)
            dy = grid_start_y + row * (die_size + street_width)
            
            # Skip dies outside wafer radius
            die_center = (dx + die_size // 2, dy + die_size // 2)
            if np.sqrt((die_center[0] - center[0])**2 + (die_center[1] - center[1])**2) > radius - 30:
                continue
                
            # Draw Scribe line / Street border
            cv2.rectangle(img, (dx - street_width//2, dy - street_width//2), 
                          (dx + die_size + street_width//2, dy + die_size + street_width//2), 45, 1)
            
            # Die Seal Ring (Outer Border)
            cv2.rectangle(img, (dx, dy), (dx + die_size, dy + die_size), 200, 2)
            cv2.rectangle(img, (dx + 3, dy + 3), (dx + die_size - 3, dy + die_size - 3), 130, 1)
            
            # Inside Die: Power Bus Lines
            # Main horizontal and vertical bus bars
            cv2.line(img, (dx + 10, dy + 25), (dx + die_size - 10, dy + 25), 220, 3)
            cv2.line(img, (dx + 10, dy + die_size - 25), (dx + die_size - 10, dy + die_size - 25), 220, 3)
            cv2.line(img, (dx + 25, dy + 10), (dx + 25, dy + die_size - 10), 220, 3)
            cv2.line(img, (dx + die_size - 25, dy + 10), (dx + die_size - 25, dy + die_size - 10), 220, 3)
            
            # Inside Die: Contact / Bond Pad Arrays (corner & perimeter pads)
            pad_positions = [
                (dx + 10, dy + 10), (dx + 35, dy + 10), (dx + 65, dy + 10), (dx + 95, dy + 10), (dx + 120, dy + 10),
                (dx + 10, dy + 120), (dx + 35, dy + 120), (dx + 65, dy + 120), (dx + 95, dy + 120), (dx + 120, dy + 120)
            ]
            for px, py in pad_positions:
                cv2.rectangle(img, (px - 4, py - 4), (px + 4, py + 4), 240, -1)
                cv2.rectangle(img, (px - 5, py - 5), (px + 5, py + 5), 160, 1)
            
            # Inside Die: Dense Micro-array / Fine Logic Lines (horizontal thin traces)
            for line_y in range(dy + 35, dy + die_size - 30, 6):
                cv2.line(img, (dx + 32, line_y), (dx + die_size - 32, line_y), 185, 1)
                
            # Inside Die: Vertical Interconnect Traces
            for line_x in range(dx + 35, dx + die_size - 35, 12):
                cv2.line(img, (line_x, dy + 35), (line_x, dy + die_size - 35), 165, 1)
                
            # Sub-module Blocks (RAM / Logic arrays)
            cv2.rectangle(img, (dx + 35, dy + 40), (dx + 65, dy + 80), 100, -1)
            cv2.rectangle(img, (dx + 35, dy + 40), (dx + 65, dy + 80), 190, 1)
            cv2.rectangle(img, (dx + 75, dy + 40), (dx + 105, dy + 80), 100, -1)
            cv2.rectangle(img, (dx + 75, dy + 40), (dx + 105, dy + 80), 190, 1)
            
            # Internal crosshatch in sub-modules
            for cy in range(dy + 44, dy + 78, 5):
                cv2.line(img, (dx + 37, cy), (dx + 63, cy), 150, 1)
                cv2.line(img, (dx + 77, cy), (dx + 103, cy), 150, 1)

    # 3. Artificial Defect Regions
    
    # Defect 1: Particle Contamination (Top-Left area ~ (335, 305))
    # Irregular dark spot with slight bright scatter halo
    particle_center = (335, 305)
    cv2.circle(img, particle_center, 12, 140, -1)
    cv2.ellipse(img, particle_center, (10, 6), 35, 0, 360, 45, -1)
    cv2.circle(img, (particle_center[0] - 2, particle_center[1] - 2), 4, 25, -1)
    cv2.circle(img, (particle_center[0] + 4, particle_center[1] + 3), 2, 230, -1)  # bright scattering highlight
    
    # Defect 2: Micro-bridge / Short Circuit (Top-Right area ~ (673, 311))
    # Unwanted conductive bridge across parallel traces
    cv2.rectangle(img, (670, 305), (676, 320), 240, -1)
    
    # Defect 3: Void / Pinhole Defect (Bottom-Left area ~ (350, 680))
    # Dark hole void inside a contact pad
    cv2.circle(img, (350, 680), 4, 40, -1)
    
    # Defect 4: Line Break / Open Circuit (Bottom-Right area ~ (685, 685))
    # Gap in a bus line trace
    cv2.rectangle(img, (680, 683), (690, 687), 75, -1)  # overwrite trace with background gray
    
    # Defect 5: Surface Scratch (Traversing across center-left ~ (400, 480) to (510, 560))
    scratch_points = np.array([
        [410, 475], [435, 495], [450, 510], [475, 525], [490, 545], [515, 565]
    ], dtype=np.int32)
    cv2.polylines(img, [scratch_points], isClosed=False, color=40, thickness=2)
    # Bright edge reflection along the scratch
    scratch_reflect = scratch_points + np.array([1, 1], dtype=np.int32)
    cv2.polylines(img, [scratch_reflect], isClosed=False, color=210, thickness=1)

    # 4. Add realistic inspection noise (Gaussian detector noise)
    np.random.seed(42)
    noise = np.random.normal(0, 3.5, (height, width)).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5. Save synthetic image
    cv2.imwrite(output_path, noisy_img)
    print(f"Synthetic semiconductor test image saved successfully at: {output_path}")
    print(f"Image shape: {noisy_img.shape}, dtype: {noisy_img.dtype}")

if __name__ == "__main__":
    generate_synthetic_semiconductor_image()
