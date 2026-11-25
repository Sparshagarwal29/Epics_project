import os
import numpy as np
from torchvision.datasets import EuroSAT

DATA_DIR = "data"
RAW_DIR = "raw_eurosat"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

print("1. Downloading official Sentinel-2 Satellite imagery via torchvision...")
dataset = EuroSAT(root=RAW_DIR, download=True)

CLASS_MAP = {
    0: 0,  
    1: 0,
    2: 0,  
    3: 1,  
    4: 2, 
    5: 0, 
    6: 0, 
    7: 2,  
    8: 3,  
    9: 3   
}

print("2. Building 128x128 multi-class Sentinel-2 spatial scenes...")
num_tiles = 500  
images_list = []
masks_list = []

n_samples = len(dataset)
np.random.seed(42)

for _ in range(num_tiles):
    indices = np.random.choice(n_samples, 4, replace=False)
    
    tile_img = np.zeros((128, 128, 3), dtype=np.uint8)
    tile_mask = np.zeros((128, 128), dtype=np.int64)
    
    quadrants = [(0, 0), (0, 64), (64, 0), (64, 64)]
    
    for idx, (r, c) in zip(indices, quadrants):
        img_pil, label = dataset[idx]
        img_np = np.array(img_pil, dtype=np.uint8)
        mapped_class = CLASS_MAP[label]
        
        tile_img[r:r+64, c:c+64, :] = img_np
        tile_mask[r:r+64, c:c+64] = mapped_class
        
    images_list.append(tile_img)
    masks_list.append(tile_mask)

images = np.array(images_list)
masks = np.array(masks_list)

print(f"Generated {len(images)} real satellite patches of size 128x128.")

n_train = int(len(images) * 0.70)
n_val = int(len(images) * 0.15)

np.save(os.path.join(DATA_DIR, "train_images.npy"), images[:n_train])
np.save(os.path.join(DATA_DIR, "train_masks.npy"), masks[:n_train])
np.save(os.path.join(DATA_DIR, "valid_images.npy"), images[n_train:n_train+n_val])
np.save(os.path.join(DATA_DIR, "valid_masks.npy"), masks[n_train:n_train+n_val])
np.save(os.path.join(DATA_DIR, "test_images.npy"), images[n_train+n_val:])
np.save(os.path.join(DATA_DIR, "test_masks.npy"), masks[n_train+n_val:])

print("Success! Real Sentinel-2 dataset arrays written to data/ folder.")