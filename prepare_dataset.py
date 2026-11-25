import os
import numpy as np

os.makedirs("data", exist_ok=True)

def generate_landcover_patches(num_samples=100, patch_size=128):
    images = np.zeros((num_samples, patch_size, patch_size, 3), dtype=np.uint8)
    masks = np.zeros((num_samples, patch_size, patch_size), dtype=np.int64)
    
    for i in range(num_samples):
        # Base background: Crops (Class 0 - Greenish)
        img = np.random.randint(20, 80, (patch_size, patch_size, 3), dtype=np.uint8)
        img[:, :, 1] = np.random.randint(120, 220, (patch_size, patch_size)) # Strong green channel
        mask = np.zeros((patch_size, patch_size), dtype=np.int64) # Class 0
        
        # Add Water Body (Class 3 - Blueish patch)
        if np.random.rand() > 0.3:
            w_start, h_start = np.random.randint(0, 64, 2)
            w_size, h_size = np.random.randint(30, 60, 2)
            img[h_start:h_start+h_size, w_start:w_start+w_size, 0] = np.random.randint(0, 50)
            img[h_start:h_start+h_size, w_start:w_start+w_size, 1] = np.random.randint(50, 100)
            img[h_start:h_start+h_size, w_start:w_start+w_size, 2] = np.random.randint(180, 255)
            mask[h_start:h_start+h_size, w_start:w_start+w_size] = 3
            
        # Add Built-up Area (Class 2 - Grayish patch)
        if np.random.rand() > 0.4:
            b_start, bh_start = np.random.randint(30, 70, 2)
            b_size, bh_size = np.random.randint(20, 45, 2)
            gray_val = np.random.randint(120, 180, (bh_size, b_size), dtype=np.uint8)
            img[bh_start:bh_start+bh_size, b_start:b_start+b_size, 0] = gray_val
            img[bh_start:bh_start+bh_size, b_start:b_start+b_size, 1] = gray_val
            img[bh_start:bh_start+bh_size, b_start:b_start+b_size, 2] = gray_val
            mask[bh_start:bh_start+bh_size, b_start:b_start+b_size] = 2

        # Add Bare Land (Class 1 - Tan/Brown patch)
        if np.random.rand() > 0.5:
            l_start, lh_start = np.random.randint(0, 50, 2)
            l_size, lh_size = np.random.randint(20, 40, 2)
            img[lh_start:lh_start+lh_size, l_start:l_start+l_size, 0] = np.random.randint(180, 220)
            img[lh_start:lh_start+lh_size, l_start:l_start+l_size, 1] = np.random.randint(140, 170)
            img[lh_start:lh_start+lh_size, l_start:l_start+l_size, 2] = np.random.randint(100, 130)
            mask[lh_start:lh_start+lh_size, l_start:l_start+l_size] = 1
            
        images[i] = img
        masks[i] = mask
        
    return images, masks

# Generate and save arrays
print("Generating dataset arrays...")
tr_img, tr_msk = generate_landcover_patches(120)
val_img, val_msk = generate_landcover_patches(30)
ts_img, ts_msk = generate_landcover_patches(30)

np.save("data/train_images.npy", tr_img)
np.save("data/train_masks.npy", tr_msk)
np.save("data/valid_images.npy", val_img)
np.save("data/valid_masks.npy", val_msk)
np.save("data/test_images.npy", ts_img)
np.save("data/test_masks.npy", ts_msk)
print("Data saved successfully into data/ folder!")