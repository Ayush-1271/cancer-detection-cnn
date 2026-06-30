# Cancer Detection CNN — Lung Histopathology

**Ayush Ranjan | 23MIP10135 | VIT Bhopal**

## Results
| Metric | Score |
|---|---|
| Test Accuracy | **96.98%** |
| Test AUC | **0.9963** |
| Top-2 Accuracy | **100.00%** |
| Parameters | 6,515,107 |

## Dataset
- **Source:** [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) — Kaggle
- **Classes:** 3 lung cancer types · **Total:** 15,000 images
- **Split:** 70% train / 15% val / 15% test
- **Image size:** 96 × 96 px

### Classes
| Class | Description |
|---|---|
| `lung_aca` | Lung Adenocarcinoma |
| `lung_n` | Normal Lung Tissue |
| `lung_scc` | Lung Squamous Cell Carcinoma |

### Per-Class Performance
| Class | Precision | Recall | F1 |
|---|---|---|---|
| lung_aca | 0.96 | 0.95 | 0.95 |
| lung_n | 1.00 | 1.00 | 1.00 |
| lung_scc | 0.95 | 0.96 | 0.96 |

## Architecture — Deep CNN (5 Conv Blocks)
```
Input (96×96×3)
├── Block 1: Conv2D(32)×2  → BatchNorm → MaxPool → Dropout(0.20)
├── Block 2: Conv2D(64)×2  → BatchNorm → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128)×2 → BatchNorm → MaxPool → Dropout(0.30)
├── Block 4: Conv2D(256)×2 → BatchNorm → MaxPool → Dropout(0.30)
├── Block 5: Conv2D(512)   → BatchNorm            → Dropout(0.30)
├── GlobalAveragePooling2D
├── Dense(1024) → BatchNorm → Dropout(0.50)
├── Dense(512)  → Dropout(0.40)
└── Dense(3, softmax)
```
All Conv layers: ReLU · same-padding · L2(1e-4) regularization

## Training Config
| Setting | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, EarlyStopping patience=8) |
| Batch Size | 64 |
| Pipeline | tf.data with image caching |
| Augmentation | rotation 90°, h/v flip, brightness, contrast |
| Framework | TensorFlow 2.20 / Keras |
| Hardware | Google Colab T4 GPU |

## How to Run
1. Open `3_Cancer_Detection_CNN.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime → **T4 GPU**
3. Run all cells top to bottom
