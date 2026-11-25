# Satellite Image Semantic Segmentation (Phase 1 Baseline)

An automated deep learning framework for multi-class Land Use / Land Cover (LULC) semantic segmentation from satellite imagery.

## 📌 Phase 1 Scope & Objectives
This repository contains Phase 1 implementation focusing on:
- Baseline U-Net (ResNet-34 backbone) pipeline.
- 4-class Land Cover patch classification (`Crops`, `Bare Land`, `Built Up`, `Water`).
- Automated data loading, patch normalization, and PyTorch dataloaders.
- Quantitative evaluation metrics: Pixel Accuracy, mIoU, and Dice Coefficient.

## 🛠️ Installation
```bash
git clone [https://github.com/your-username/satellite-lulc-segmentation.git](https://github.com/your-username/satellite-lulc-segmentation.git)
cd satellite-lulc-segmentation
pip install -r requirements.txt