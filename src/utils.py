import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = ["Crops", "Bare", "Built", "Water"]
CLASS_COLORS = [
    (0, 255, 0),      # Crops - Green
    (210, 180, 140),  # Bare - Tan
    (128, 128, 128),  # Built - Gray
    (0, 0, 255),      # Water - Blue
]

def colorize_mask(mask):
    """Convert class-index mask (H, W) to RGB mask (H, W, 3)."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(CLASS_COLORS):
        color_mask[mask == class_idx] = color
    return color_mask

def plot_sample_prediction(image, label, pred):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image)
    axs[0].set_title("Input Satellite Patch")
    axs[0].axis('off')

    axs[1].imshow(colorize_mask(label))
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(colorize_mask(pred))
    axs[2].set_title("U-Net Prediction")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()