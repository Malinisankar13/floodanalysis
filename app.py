import os
import io
import base64
import torch
import numpy as np
import time
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from models.flood_eye_cnn import FloodEyeCNN
from models.flood_lstm import FloodLSTM
import torchvision.transforms as T
from utils.metrics import (
    pixel_accuracy, miou, dice_coeff, flooded_area, 
    get_cnn_metrics, get_lstm_metrics
)

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "saved_models"
DATA_DIR = "data"

# Load models
cnn_model = FloodEyeCNN().to(DEVICE)
cnn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "flood_eye_cnn.pth"), map_location=DEVICE))
cnn_model.eval()

# LSTM expects input_size=1 (it predicts area from sequence of areas)
lstm_model = FloodLSTM(input_size=1).to(DEVICE)
lstm_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "flood_lstm.pth"), map_location=DEVICE))
lstm_model.eval()

# Metric Configuration
PIXELS_TO_SQM = 0.25  # Assumption: 1 pixel = 0.25 m^2 (0.5m GSD)

# Color Mapping (R, G, B)
CLASS_COLORS = {
    0: (0, 0, 0),       # Background -> Black
    1: (255, 60, 60),   # Building   -> Red
    2: (255, 255, 60),  # Road       -> Yellow
    3: (60, 100, 255),  # Water      -> Blue
    4: (60, 255, 255)   # Pool       -> Cyan
}

def find_ground_truth(filename):
    """Attempt to find ground truth mask for a given filename."""
    if not filename: return None
    base = os.path.splitext(filename)[0]
    mask_dirs = [
        os.path.join(DATA_DIR, "raw", "train", "train-label-img"),
        os.path.join(DATA_DIR, "raw", "test", "label"),
    ]
    candidates = [f"{base}_lab.png", f"{base}.png", filename]
    for m_dir in mask_dirs:
        if not os.path.isdir(m_dir): continue
        for cand in candidates:
            path = os.path.join(m_dir, cand)
            if os.path.exists(path):
                return path
    return None

def process_image(img_bytes, filename=None):
    start_time = time.time()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        seg_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        # 1. Create Color-Mapped Mask
        h, w = seg_mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, color in CLASS_COLORS.items():
            color_mask[seg_mask == class_idx] = color
            
        mask_img = Image.fromarray(color_mask)
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 2. Calculate Real-World Metrics
        class_stats = {}
        total_flood_pixels = 0
        names = {1: "Building", 2: "Road", 3: "Water", 4: "Pool"}
        
        for idx, name in names.items():
            pixels = int(np.sum(seg_mask == idx))
            total_flood_pixels += pixels
            area_sqm = pixels * PIXELS_TO_SQM
            class_stats[name] = {
                "pixels": pixels,
                "area_m2": round(area_sqm, 2),
                "area_km2": round(area_sqm / 1_000_000, 6)
            }
            
        total_area_sqm = total_flood_pixels * PIXELS_TO_SQM

        # 3. Calculate Advanced Metrics if Ground Truth exists
        quality_metrics = {
            "accuracy": 0.0, "iou": 0.0, "dice": 0.0, 
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "has_gt": False, "gt_area_m2": 0.0
        }
        gt_path = find_ground_truth(filename)
        
        if gt_path:
            gt_img = Image.open(gt_path).convert("L")
            gt_tensor = T.Compose([
                T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor()
            ])(gt_img)
            
            gt_np = torch.round(gt_tensor * 255).squeeze().numpy()
            multi_gt = torch.zeros((128, 128), dtype=torch.long)
            multi_gt[gt_np == 2] = 1 # Building
            multi_gt[gt_np == 4] = 2 # Road
            multi_gt[gt_np == 5] = 3 # Water
            multi_gt[gt_np == 8] = 4 # Pool
            
            multi_gt_tensor = multi_gt.unsqueeze(0).to(DEVICE)
            
            # Legacy metrics
            quality_metrics["accuracy"] = round(pixel_accuracy(outputs, multi_gt_tensor), 4)
            quality_metrics["iou"] = round(miou(outputs, multi_gt_tensor), 4)
            quality_metrics["dice"] = round(dice_coeff(outputs, multi_gt_tensor), 4)
            
            # New Advanced metrics
            adv = get_cnn_metrics(outputs, multi_gt_tensor)
            quality_metrics.update({
                "precision": round(adv["precision"], 4),
                "recall": round(adv["recall"], 4),
                "f1": round(adv["f1_score"], 4),
                "confusion_matrix": adv["confusion_matrix"],
                "has_gt": True
            })
            
            gt_flood_pixels = int(torch.sum(multi_gt > 0).item())
            quality_metrics["gt_area_m2"] = round(gt_flood_pixels * PIXELS_TO_SQM, 2)
        
    processing_time = time.time() - start_time
    return {
        "total_pixels": total_flood_pixels,
        "total_m2": round(total_area_sqm, 2),
        "total_km2": round(total_area_sqm / 1_000_000, 6),
        "breakdown": class_stats,
        "mask_b64": mask_base64,
        "quality": quality_metrics,
        "processing_time": round(processing_time * 1000, 2) # in ms
    }

@app.post("/api/process")
async def process_sequence(files: List[UploadFile] = File(...)):
    start_time_total = time.time()
    results = []
    for file in files:
        content = await file.read()
        res = process_image(content, filename=file.filename)
        results.append(res)
    
    # LSTM Prediction
    areas = [r["total_m2"] for r in results]
    
    predicted_area_m2 = 0
    lstm_eval = {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "has_eval": False}
    
    if len(areas) >= 2:
        # We use all but the last one to predict the last one if GT exists for the last one
        # Or just predict the next step. 
        # To show MAE/RMSE, we'll try to predict the LAST image using previous ones.
        
        # 1. Predict real T_next
        recent_areas = areas[-4:]
        while len(recent_areas) < 4: recent_areas.insert(0, recent_areas[0])
        norm_factor = 128 * 128 * PIXELS_TO_SQM
        x = torch.tensor(recent_areas).float().view(1, 4, 1) / norm_factor
        
        with torch.no_grad():
            pred = lstm_model(x.to(DEVICE)).cpu().item()
            predicted_area_m2 = round(pred * norm_factor, 2)
            
        # 2. Evaluate LSTM if we have ground truth for the last step
        # Let's see if the last image in sequence has GT.
        last_res = results[-1]
        if last_res["quality"]["has_gt"]:
            # Actual GT area
            actual_gt_area = last_res["quality"]["gt_area_m2"]
            
            # Predict the last one using images [0:last-1]
            if len(areas) >= 2:
                seq_for_eval = areas[:-1]
                if len(seq_for_eval) > 0:
                    while len(seq_for_eval) < 4: seq_for_eval.insert(0, seq_for_eval[0])
                    x_eval = torch.tensor(seq_for_eval[-4:]).float().view(1, 4, 1) / norm_factor
                    with torch.no_grad():
                        pred_val = lstm_model(x_eval.to(DEVICE)).cpu().item() * norm_factor
                        lstm_eval.update(get_lstm_metrics([actual_gt_area], [pred_val]))
                        lstm_eval["has_eval"] = True
    else:
        predicted_area_m2 = areas[0] if areas else 0
        
    total_time = (time.time() - start_time_total) * 1000 # ms
        
    return {
        "status": "Success",
        "count": len(results),
        "areas": [r["total_pixels"] for r in results],
        "metrics_m2": areas,
        "breakdowns": [r["breakdown"] for r in results],
        "masks": [r["mask_b64"] for r in results],
        "quality_metrics": [r["quality"] for r in results],
        "lstm_metrics": lstm_eval,
        "prediction_m2": predicted_area_m2,
        "timing": {
            "total_ms": round(total_time, 2),
            "avg_per_image_ms": round(total_time / len(results), 2) if results else 0
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
