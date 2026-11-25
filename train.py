import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_unet_model
from src.metrics import compute_mIoU, compute_dice

# Configuration
DATA_DIR = "./data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
PATIENCE = 8
LR = 6e-5
NUM_CLASSES = 4

def train_pipeline():
    train_loader, val_loader, test_loader = get_dataloaders(
        os.path.join(DATA_DIR, "train_images.npy"),
        os.path.join(DATA_DIR, "train_masks.npy"),
        os.path.join(DATA_DIR, "valid_images.npy"),
        os.path.join(DATA_DIR, "valid_masks.npy"),
        os.path.join(DATA_DIR, "test_images.npy"),
        os.path.join(DATA_DIR, "test_masks.npy"),
    )

    model = build_unet_model(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(pixel_values)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_unet_model.pt")
            epochs_no_improve = 0
            print("--> Model checkpoint saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Final Evaluation on Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    model.load_state_dict(torch.load("best_unet_model.pt"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(pixel_values)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    iou_per_class, mean_iou = compute_mIoU(all_preds, all_labels, NUM_CLASSES)
    dice_per_class, mean_dice = compute_dice(all_preds, all_labels, NUM_CLASSES)

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f}")

if __name__ == "__main__":
    train_pipeline()