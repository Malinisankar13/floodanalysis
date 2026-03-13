import torch
import numpy as np

def pixel_accuracy(pred, target):
    # pred is (B, C, H, W) logits/probs, target is (B, H, W) long
    pred_classes = pred.argmax(1)
    correct = (pred_classes == target).float().sum()
    total = torch.numel(target)
    return (correct / total).item()

def miou(pred, target, num_classes=5):
    pred_classes = pred.argmax(1)
    ious = []
    for cls in range(num_classes):
        inter = ((pred_classes == cls) & (target == cls)).sum().float()
        union = ((pred_classes == cls) | (target == cls)).sum().float()
        if union == 0:
            ious.append(torch.tensor(1.0, device=pred.device))
        else:
            ious.append(inter / union)
    return torch.mean(torch.stack(ious)).item()

def dice_coeff(pred, target, num_classes=5):
    pred_classes = pred.argmax(1)
    dice_scores = []
    for cls in range(num_classes):
        inter = ((pred_classes == cls) & (target == cls)).sum().float()
        total_pixels = (pred_classes == cls).sum().float() + (target == cls).sum().float()
        if total_pixels == 0:
            dice_scores.append(torch.tensor(1.0, device=pred.device))
        else:
            dice_scores.append((2.0 * inter) / total_pixels)
    return torch.mean(torch.stack(dice_scores)).item()

def get_cnn_metrics(pred, target, num_classes=5):
    """Calculate Precision, Recall, F1 and Confusion Matrix for CNN."""
    pred_classes = pred.argmax(1).flatten()
    target_flat = target.flatten()
    
    # Confusion Matrix
    cm = torch.zeros((num_classes, num_classes), device=pred.device)
    for t, p in zip(target_flat, pred_classes):
        cm[t.long(), p.long()] += 1
    
    # Precision, Recall, F1 per class
    precision = []
    recall = []
    f1 = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(1.0, device=pred.device)
        rec = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(1.0, device=pred.device)
        f = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else torch.tensor(1.0, device=pred.device)
        
        precision.append(prec)
        recall.append(rec)
        f1.append(f)
        
    return {
        "precision": torch.mean(torch.stack(precision)).item(),
        "recall": torch.mean(torch.stack(recall)).item(),
        "f1_score": torch.mean(torch.stack(f1)).item(),
        "confusion_matrix": cm.cpu().numpy().tolist()
    }

def get_lstm_metrics(y_true, y_pred):
    """MAE, RMSE, MAPE for LSTM."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if np.any(mask) else 0.0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }

def flooded_area(mask):
    return (mask > 0).sum().item()
