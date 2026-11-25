import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import build_unet_model
from src.utils import colorize_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_prediction():
    test_images = np.load("data/test_images.npy")
    test_masks = np.load("data/test_masks.npy")
    
    model = build_unet_model(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load("best_unet_model.pt"))
    model.eval()

    img_raw = test_images[0]
    mask_gt = test_masks[0]

    img_tensor = torch.tensor(img_raw.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_raw)
    axs[0].set_title("Input Satellite Patch")
    axs[0].axis("off")

    axs[1].imshow(colorize_mask(mask_gt))
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis("off")

    axs[2].imshow(colorize_mask(pred_mask))
    axs[2].set_title("U-Net Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    output_path = "prediction_result.png"
    plt.savefig(output_path, dpi=300)
    print(f"Visual prediction saved to {output_path}!")

if __name__ == "__main__":
    run_prediction()