import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from models.flood_eye_cnn import FloodEyeCNN
from models.flood_lstm import FloodLSTM
from utils.metrics import flooded_area
from utils.visualizer import plot_sequence, plot_flood_analysis

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
SIZE = (128, 128)

def load_models():
    cnn = FloodEyeCNN().to(DEVICE)
    lstm = FloodLSTM().to(DEVICE)
    cnn_path = os.path.join(MODEL_DIR, "flood_eye_cnn.pth")
    lstm_path = os.path.join(MODEL_DIR, "flood_lstm.pth")
    
    if os.path.exists(cnn_path):
        cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
    
    cnn.eval()
    lstm.eval()
    return cnn, lstm

def predict_single(cnn_model, img_path):
    img = Image.open(img_path).convert("RGB")
    tf = T.Compose([T.Resize(SIZE), T.ToTensor()])
    img_tensor = tf(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = cnn_model(img_tensor)
        mask = output.argmax(1).cpu().numpy()[0]
    
    area = flooded_area(mask)
    return mask, area

def run_pipeline(sequence_paths):
    cnn, lstm = load_models()
    
    areas = []
    images = []
    masks = []
    titles = []
    
    print(f"{'Step':<10} | {'Area (px)':<10} | {'Status':<15}")
    print("-" * 40)
    
    for i, path in enumerate(sequence_paths):
        mask, area = predict_single(cnn, path)
        areas.append(area)
        
        # Collect for visualization
        images.append(np.array(Image.open(path).convert("RGB")))
        masks.append(mask)
        titles.append(f"Step {i+1}")
        
        status = "Normal"
        # Simple anomaly detection: sudden jump > 15% of image size
        if i > 0:
            diff = areas[i] - areas[i-1]
            if diff > (SIZE[0] * SIZE[1] * 0.15):
                status = "[!] ANOMALY!"
        
        print(f"Step {i+1:<5} | {area:<10} | {status}")
    
    # Predict next area using LSTM
    pred_area = None
    if len(areas) >= 4:
        seq_tensor = torch.tensor(areas[-4:]).float().view(1, 4, 1)
        seq_tensor = seq_tensor / (SIZE[0] * SIZE[1]) # Normalize
        
        with torch.no_grad():
            pred_norm = lstm(seq_tensor.to(DEVICE)).item()
        
        pred_area = int(pred_norm * SIZE[0] * SIZE[1])
        print("-" * 40)
        print(f"[*] Predicted future flood area: {pred_area} px")

    # Generate Visualizations
    OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    plot_sequence(images, masks, titles, save_path=os.path.join(OUT_DIR, "sequence_vis.png"))
    plot_flood_analysis(areas, prediction=pred_area, save_path=os.path.join(OUT_DIR, "trend_analysis.png"))

if __name__ == "__main__":
    # Test with real images from the dataset
    DATA_ORG = os.path.join(os.path.dirname(__file__), "data", "raw", "train", "train-org-img")
    # Picking a different set of sample images manually for verification
    test_files = ["10817.jpg", "10818.jpg", "10819.jpg", "10820.jpg", "10821.jpg"]
    test_seq = [os.path.join(DATA_ORG, f) for f in test_files if os.path.exists(os.path.join(DATA_ORG, f))]
    
    if len(test_seq) > 0:
        run_pipeline(test_seq)
    else:
        print(f"Data not found in {DATA_ORG}. Please verify paths.")
