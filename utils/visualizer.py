import matplotlib.pyplot as plt
import numpy as np
import os

def overlay(image, mask, alpha=0.5):
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha, cmap="jet")
    plt.axis("off")
    plt.show()

def plot_sequence(images, masks, titles, save_path=None):
    """Plots a sequence of images and their corresponding masks."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(titles[i])
        axes[0, i].axis("off")
        
        axes[1, i].imshow(masks[i], cmap="jet")
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Sequence visualization saved to {save_path}")
    else:
        plt.show()

def plot_flood_analysis(areas, prediction=None, save_path=None):
    """Plots the flood area trend and prediction."""
    plt.figure(figsize=(10, 6))
    time_steps = list(range(1, len(areas) + 1))
    plt.plot(time_steps, areas, marker="o", linestyle="-", color="b", label="Detected Area")
    
    if prediction is not None:
        next_step = len(areas) + 1
        plt.plot([len(areas), next_step], [areas[-1], prediction], 
                 marker="s", linestyle="--", color="r", label="Predicted Future")
        plt.scatter(next_step, prediction, color="r")
    
    plt.title("Flood Area Expansion Analysis")
    plt.xlabel("Time Step")
    plt.ylabel("Area (pixels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trend analysis saved to {save_path}")
    else:
        plt.show()
