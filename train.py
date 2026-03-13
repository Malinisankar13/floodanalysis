import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.flood_eye_cnn import FloodEyeCNN
from models.flood_lstm import FloodLSTM
import numpy as np
from utils.dataset import FloodDataset
from utils.metrics import pixel_accuracy, miou, flooded_area

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_CNN = 1
EPOCHS_LSTM = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_cnn(): 
    print("--- Training CNN (Multi-Class) ---")
    train_dir = os.path.join(DATA_DIR, "raw", "train")
    dataset = FloodDataset(train_dir, train_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FloodEyeCNN(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS_CNN):
        model.train()
        total_loss = 0
        for i, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS_CNN}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}", flush=True)
            if (i + 1) >= 80:
                print("Demo training target reached. Saving...", flush=True)
                break
        print(f"Epoch {epoch+1}/{EPOCHS_CNN} Complete, Average Loss: {total_loss/len(loader):.4f}", flush=True)
    
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "flood_eye_cnn.pth"))
    return model

def extract_areas(cnn_model):
    print("--- Extracting Multi-Class Areas for LSTM ---")
    cnn_model.eval()
    train_dir = os.path.join(DATA_DIR, "raw", "train")
    dataset = FloodDataset(train_dir, train_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_areas = [] # List of total flooded areas (sum of classes 1,2,3,4)
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            outputs = cnn_model(imgs)
            pred = outputs.argmax(1).cpu().numpy()[0]
            
            # Total flood area is sum of all non-zero classes
            flood_pixels = np.sum(pred > 0)
            all_areas.append(flood_pixels)
    
    # all_areas is a flat list, but we know each sequence is 5 steps
    sequences = []
    for i in range(0, len(all_areas), 5):
        sequences.append(all_areas[i:i+5])
    return sequences

def train_lstm(sequences):
    print("--- Training LSTM ---")
    # Prepare data: use first 4 steps to predict 5th
    X, y = [], []
    for seq in sequences:
        if len(seq) == 5:
            X.append(seq[:4])
            y.append(seq[4])
    
    X = torch.tensor(X).float().unsqueeze(-1) # (B, 4, 1)
    y = torch.tensor(y).float().unsqueeze(-1) # (B, 1)
    
    # Normalize (simple scaling by image size)
    X = X / (128 * 128)
    y = y / (128 * 128)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = FloodLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS_LSTM):
        model.train()
        total_loss = 0
        for i, (bx, by) in enumerate(loader):
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"LSTM Epoch {epoch+1}/{EPOCHS_LSTM}, Batch {i+1}/{len(loader)}, Loss: {loss.item():.6f}", flush=True)
        # print(f"Epoch {epoch+1}/{EPOCHS_LSTM}, Loss: {total_loss/len(loader):.6f}", flush=True) # Redundant with above change

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "flood_lstm.pth"))

if __name__ == "__main__":
    cnn = train_cnn()
    seqs = extract_areas(cnn)
    train_lstm(seqs)
    print("Training complete. Models saved in saved_models/", flush=True)
