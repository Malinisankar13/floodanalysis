import os
import numpy as np
from PIL import Image

mask_dir = r'c:\Users\malin\Desktop\flood_analysis\data\raw\train\train-label-img'
results = {}

for f in sorted(os.listdir(mask_dir)):
    if f.startswith('seq_') and f.endswith('.png'):
        seq_id = f.split('_t')[0]
        mask_path = os.path.join(mask_dir, f)
        img = Image.open(mask_path).convert('L')
        # In my mock_gen.py, water is class 5
        area = np.sum(np.array(img) == 5)
        if seq_id not in results:
            results[seq_id] = []
        results[seq_id].append((f, area))

print("--- Sequence Analysis (Increase / Decrease) ---")
for seq, data in results.items():
    areas = [d[1] for d in data]
    diffs = [areas[i] - areas[i-1] for i in range(1, len(areas))]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    
    status = "Stable"
    if avg_diff > 500: status = "INCREASE (Expansion)"
    elif avg_diff < -500: status = "DECREASE (Recession)"
    
    print(f"{seq}: {status} -> {[d[0].replace('_lab.png', '.png') for d in data]}")

print("\n--- Potential 'No Major Change' Pairs (from Real Dataset) ---")
# Find pairs of real images with similar water area
real_data = []
real_mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('seq_')])
for f in real_mask_files[:100]:
    mask_path = os.path.join(mask_dir, f)
    img = Image.open(mask_path).convert('L')
    area = np.sum(np.array(img) == 5)
    real_data.append((f.replace('_lab.png', '.jpg'), area))

for i in range(len(real_data) - 1):
    img1, a1 = real_data[i]
    img2, a2 = real_data[i+1]
    if a1 > 0 and abs(a1 - a2) < 1000: # Significant water but similar
        print(f"PAIR: {img1} and {img2} (Area Difference: {abs(a1-a2)})")
        break
