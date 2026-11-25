import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LULCSegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Scale image pixels to [0, 1] and reshape to (C, H, W)
        image = self.images[idx].astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        
        # Target mask tensor
        mask = torch.tensor(self.masks[idx], dtype=torch.long)
        
        return {'pixel_values': image, 'labels': mask}

def get_dataloaders(train_img_path, train_mask_path, val_img_path, val_mask_path, test_img_path, test_mask_path, batch_size=32):
    train_images, train_masks = np.load(train_img_path), np.load(train_mask_path)
    val_images, val_masks     = np.load(val_img_path), np.load(val_mask_path)
    test_images, test_masks   = np.load(test_img_path), np.load(test_mask_path)

    train_ds = LULCSegmentationDataset(train_images, train_masks)
    val_ds   = LULCSegmentationDataset(val_images, val_masks)
    test_ds  = LULCSegmentationDataset(test_images, test_masks)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader