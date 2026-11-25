# Satellite Image Semantic Segmentation (Phase 1 Baseline)

An automated deep learning framework for multi-class Land Use / Land Cover (LULC) semantic segmentation from real satellite imagery, developed as part of the regional geospatial environmental monitoring initiative (Sehore / Kothrikalan LULC analysis).

---

## 📌 Project Overview & Phase 1 Scope

This repository contains the **Phase 1 (~30–35% project scope)** baseline implementation. The objective of Phase 1 is to construct a production-ready, modular PyTorch engineering framework capable of processing multi-spectral/optical satellite image patches and performing dense pixel-level classification across four primary land cover categories.

### Key Objectives Achieved in Phase 1:
- **Modular Software Architecture:** Clean separation of concerns into dataset management, network definition, evaluation metrics, and utility functions (`src/`).
- **Real Satellite Data Ingestion:** Integration of ESA Sentinel-2 imagery (via EuroSAT multi-class patch assembly) mapped to project land cover targets.
- **Deep Learning Baseline:** Implementation of a U-Net semantic segmentation network with a ResNet-34 encoder backbone.
- **Quantitative Metrics Pipeline:** GPU-accelerated calculation of Mean Intersection over Union ($mIoU$), Mean Dice Coefficient, and Overall Pixel Accuracy.
- **Automated Checkpointing & Inference:** Best-model state saving (`best_unet_model.pt`) and visual inference script exporting comparison figures (`prediction_result.png`).

---

## 🏷️ Land Cover Target Classes

The segmentation model maps satellite pixel patches into four distinct LULC classes aligned with regional satellite analysis requirements:

| Class Index | Class Name | Description | Color Code |
| :---: | :--- | :--- | :---: |
| **0** | **Crops / Vegetation** | Agricultural land, crop fields, forest cover, and dense canopy | Green |
| **1** | **Bare Land / Soil** | Uncultivated land, exposed soil, rocky terrain, and dirt roads | Tan / Brown |
| **2** | **Built-Up** | Residential areas, concrete structures, industrial buildings, and asphalt | Gray |
| **3** | **Water** | Water reservoirs, lakes, rivers, and seasonal water bodies | Blue |

---

## 📁 Repository Directory Structure

```text
Epics/
│
├── data/                      # Processed binary dataset splits (.npy files - GitIgnored)
│   ├── train_images.npy
│   ├── train_masks.npy
│   ├── valid_images.npy
│   ├── valid_masks.npy
│   ├── test_images.npy
│   └── test_masks.npy
│
├── raw_eurosat/               # Raw Sentinel-2 satellite data downloads (GitIgnored)
├── notebook/                  # Experimental Jupyter notebooks
│
├── src/                       # Core Python Source Package
│   ├── __init__.py
│   ├── dataset.py             # Custom PyTorch Dataset class with normalization
│   ├── model.py               # U-Net architecture definition with ResNet-34 backbone
│   ├── metrics.py             # Multi-class mIoU, Dice, and Accuracy calculators
│   └── utils.py               # Mask colorization and visualization helpers
│
├── fetch_real_data.py         # Satellite dataset fetching and patch preprocessor
├── train.py                   # Model training and validation execution pipeline
├── predict.py                 # Visual evaluation and inference generator
│
├── .gitignore                 # Excludes raw data, large model checkpoints, & virtual environment
├── best_unet_model.pt         # Saved PyTorch model checkpoint (GitIgnored)
├── prediction_result.png      # Output evaluation image (GitIgnored)
├── requirements.txt           # Python dependency requirements
└── README.md                  # Project documentation
```

---

## ⚙️ Model Architecture & Evaluation Metrics

### Network Backbone
- **Architecture:** U-Net (Encoder-Decoder with skip connections)
- **Encoder:** Pretrained ResNet-34 backbone for hierarchical feature extraction
- **Input Channels:** 3 (RGB Satellite Spectral Bands)
- **Output Classes:** 4 (Pixel-wise class probability maps via Softmax)

### Evaluation Metrics
The pipeline evaluates model performance using standard remote sensing metrics:
1. **Intersection over Union ($IoU$):**
   $$IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}$$
2. **Mean IoU ($mIoU$):**
   $$mIoU = \frac{1}{N} \sum_{c=1}^{N} IoU_c$$
3. **Dice Coefficient (F1 Score):**
   $$Dice_c = \frac{2 \cdot TP_c}{2 \cdot TP_c + FP_c + FN_c}$$

---

## 🛠️ Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/Sparshagarwal29/Epics_project.git
cd satellite-lulc-segmentation


python -m venv venv

source venv/Scripts/activate


source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution Workflow

### Step 1: Fetch and Preprocess Real Satellite Data
Download official Sentinel-2 imagery patches and build the $128 \times 128$ spatial arrays:

```bash
python fetch_real_data.py
```
*Output: Populates `data/` directory with 6 `.npy` files for Train, Validation, and Test splits.*

### Step 2: Train the U-Net Baseline Model
Execute the PyTorch training loop with automatic validation tracking and checkpointing:

```bash
python train.py
```
*Output: Saves the best performing model weights to `best_unet_model.pt` based on validation loss.*

### Step 3: Run Inference & Generate Visual Outputs
Evaluate the trained baseline model on unseen test patches and export comparison plots:

```bash
python predict.py
```
*Output: Generates `prediction_result.png` displaying Input Satellite Patch, Ground Truth Mask, and U-Net Prediction.*

---

## 📅 Next Steps & Phase 2 Roadmap

- [ ] **Multi-spectral Band Stacking:** Expand input depth from 3-channel RGB to 13-band Sentinel-2 spectral data (including NIR and SWIR bands for water/vegetation indices).
- [ ] **Advanced Backbones:** Benchmark U-Net++ and Feature Pyramid Networks (FPN).
- [ ] **Data Augmentation:** Integrate geospatial augmentations (random flips, rotations, scale jitters) via `Albumentations`.
- [ ] **Site-Specific GeoTIFF Validation:** Process cropped GeoTIFF tiles directly over the Sehore regional study area.