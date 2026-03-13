import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class FloodDataset(Dataset):
    """
    Dataset loader for FloodNet-style data.
    Supports both flat structure (img_dir/masks) and nested structure (train/train-org-img, train/train-label-img).
    """
    def __init__(self, img_dir, mask_dir, size=(128, 128)):
        self.size = size
        
        # Check if using nested FloodNet structure
        if os.path.isdir(os.path.join(img_dir, "train-org-img")):
            # Nested structure: data/raw/train/train-org-img
            self.img_dir = os.path.join(img_dir, "train-org-img")
            self.mask_dir = os.path.join(img_dir, "train-label-img")
        elif os.path.isdir(os.path.join(img_dir, "train")):
            # One level up: data/raw/train
            self.img_dir = os.path.join(img_dir, "train", "train-org-img")
            self.mask_dir = os.path.join(img_dir, "train", "train-label-img")
        else:
            # Flat structure: img_dir and mask_dir are direct paths
            self.img_dir = img_dir
            self.mask_dir = mask_dir
        
        # Get image files (support both .jpg and .png)
        self.imgs = sorted([f for f in os.listdir(self.img_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Build mask mapping: image_name -> mask_name
        mask_files = os.listdir(self.mask_dir)
        self.mask_map = {}
        for img in self.imgs:
            base = os.path.splitext(img)[0]
            # Try different mask naming conventions
            candidates = [
                f"{base}_lab.png",      # FloodNet style: 6279.jpg -> 6279_lab.png
                f"{base}.png",          # Same name, different ext
                img,                    # Same name
            ]
            for c in candidates:
                if c in mask_files:
                    self.mask_map[img] = c
                    break
        
        # Filter to only images with masks
        self.imgs = [img for img in self.imgs if img in self.mask_map]
        print(f"Loaded {len(self.imgs)} image-mask pairs from {self.img_dir}")
        
        self.tf = T.Compose([
            T.Resize(size),
            T.ToTensor()
        ])
        self.mask_tf = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        mask_name = self.mask_map[img_name]
        
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        
        img_tensor = self.tf(img)
        mask_tensor = self.mask_tf(mask)
        
        # Multi-class mapping based on FloodNet
        # 0: Background
        # 1: Building (Flooded) -> FloodNet 2
        # 2: Road (Flooded)     -> FloodNet 4
        # 3: Water              -> FloodNet 5
        # 4: Pool               -> FloodNet 8
        
        mask_np = torch.round(mask_tensor * 255).squeeze().numpy()
        multi_mask = torch.zeros((self.size[1], self.size[0]), dtype=torch.long)
        
        multi_mask[mask_np == 2] = 1 # Building
        multi_mask[mask_np == 4] = 2 # Road
        multi_mask[mask_np == 5] = 3 # Water
        multi_mask[mask_np == 8] = 4 # Pool
        
        return img_tensor, multi_mask
