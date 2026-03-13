"""
Generate synthetic UAV-like images and masks for quick testing.
Usage: python mock_gen.py
"""
import os
import numpy as np
from PIL import Image, ImageDraw

OUT_RAW = os.path.join(os.path.dirname(__file__), "raw")
OUT_MASKS = os.path.join(os.path.dirname(__file__), "masks")
os.makedirs(OUT_RAW, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)

def gen_sequence(seq_id, num_steps=5, size=256):
    """Generates a sequence of images/masks showing flood expansion or recession."""
    # Base terrain (static green)
    base_color = (34, 139, 34)
    
    # 50% chance of expansion, 50% chance of recession
    is_expansion = np.random.random() < 0.5
    
    # Random starting blob parameters
    if is_expansion:
        center_x, center_y = np.random.randint(50, size-50, size=2)
        radius = np.random.randint(10, 30)
        growth = np.random.randint(10, 20)
    else:
        center_x, center_y = np.random.randint(size//3, 2*size//3, size=2)
        radius = np.random.randint(80, 100) # Start large for recession
        growth = -np.random.randint(10, 20) # Shrinkage
    
    # Anomaly chance (20%)
    has_anomaly = np.random.random() < 0.2
    anomaly_step = np.random.randint(1, num_steps) if has_anomaly else -1

    for t in range(num_steps):
        img = Image.new("RGB", (size, size), base_color)
        mask = Image.new("L", (size, size), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        # Apply growth/decay
        current_radius = radius + t * growth
        if t == anomaly_step:
            if is_expansion:
                current_radius += 50  # Sudden spike
            else:
                current_radius -= 30  # Sudden drop
        
        current_radius = max(5, current_radius) # Keep a minimum size

        # Draw water blob (using class 3 color from Class Colors in app.py)
        # Class 3: (60, 100, 255) - Blue
        coords = [center_x - current_radius, center_y - current_radius, 
                  center_x + current_radius, center_y + current_radius]
        draw_img.ellipse(coords, fill=(60, 100, 255))
        draw_mask.ellipse(coords, fill=5) # Using 5 for Water to match train.py threshold or mapping

        img_name = f"seq_{seq_id:02d}_t{t:02d}.png"
        img.save(os.path.join(OUT_RAW, img_name))
        mask.save(os.path.join(OUT_MASKS, img_name))

if __name__ == "__main__":
    # Generate 20 sequences of 5 steps each
    for i in range(20):
        gen_sequence(i)
    print("Sequential synthetic data generated in data/raw and data/masks")
