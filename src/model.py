import torch
import segmentation_models_pytorch as smp

def build_unet_model(num_classes=4, encoder_name="resnet34"):
    model = smp.Unet(
        encoder_name=encoder_name,
        in_channels=3,
        classes=num_classes
    )
    return model