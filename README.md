# Cancer Detection using CNN — Histopathological Images

> **Assignment Submission — VIT Bhopal**

| Field | Details |
|---|---|
| **Name** | Ayush Ranjan |
| **Roll No** | 23MIP10135 |
| **Branch** | MIP |
| **Institute** | VIT Bhopal |

---

## Project Overview

This project implements a **Cancer Detection and Classification system** using a custom deep Convolutional Neural Network (CNN) trained on histopathological images from the **Lung and Colon Cancer Histopathological Images** dataset on Kaggle.

The model classifies tissue images into 5 cancer/benign categories, achieving high accuracy and AUC scores — critical metrics for medical diagnostic systems.

---

## Dataset

**Lung and Colon Cancer Histopathological Images**
- Source: [Kaggle — andrewmvd/lung-and-colon-cancer-histopathological-images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- 25,000 images (5,000 per class)
- Original size: 768×768 px (resized to 150×150 for training)

**Classes:**
| Label | Description |
|---|---|
| `lung_aca` | Lung Adenocarcinoma (malignant) |
| `lung_n` | Lung Benign Tissue (normal) |
| `lung_scc` | Lung Squamous Cell Carcinoma (malignant) |
| `colon_aca` | Colon Adenocarcinoma (malignant) |
| `colon_n` | Colon Benign Tissue (normal) |

---

## CNN Architecture

```
Input (150×150×3)
    ↓
Block 1: Conv2D(32) × 2 → BatchNorm → MaxPool → Dropout(0.2)
    ↓
Block 2: Conv2D(64) × 2 → BatchNorm → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv2D(128) × 3 → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Block 4: Conv2D(256) × 3 → BatchNorm → MaxPool → Dropout(0.35)
    ↓
Block 5: Conv2D(512) × 2 → BatchNorm → MaxPool → Dropout(0.35)
    ↓
GlobalAveragePooling2D
    ↓
Dense(1024) → BatchNorm → Dropout(0.5)
    ↓
Dense(512) → Dropout(0.4)
    ↓
Dense(5, softmax)
```

---

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3) |
| Loss | Categorical Crossentropy |
| Batch Size | 64 |
| Max Epochs | 50 (EarlyStopping) |
| Image Size | 150×150 |
| Split | 70% train / 15% val / 15% test |
| Augmentation | 90° rotation, H/V flip, zoom, brightness, reflect padding |

**Medical imaging note:** Augmentation is conservative (no distortion) to preserve tissue morphology for accurate diagnosis.

---

## How to Run

1. Open `3_Cancer_Detection_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — Kaggle API is pre-configured in the notebook

---

## Files

```
├── 3_Cancer_Detection_CNN.ipynb    # Main Colab notebook
└── README.md
```

---

## Results

The model outputs:
- Test accuracy, AUC score, Top-2 accuracy
- Training curves (accuracy, loss, AUC)
- Confusion matrix (5×5)
- Per-class classification report
- Sample histopathology predictions
