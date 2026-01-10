# Insulator_System_m: Advanced Insulator Detection based on YOLOv11

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  &nbsp;&nbsp;
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/ultralytics/ultralytics">
    <img src="https://img.shields.io/badge/Framework-Ultralytics-red.svg" alt="Framework: Ultralytics">
  </a>
</p>

This repository is dedicated to insulator detection and segmentation tasks using the **YOLOv11** framework. It incorporates several state-of-the-art (SOTA) architectural improvements, including Multi-Scale Context Blocks (MSCB), Wavelet Transform Convolutions (WTConv), and Mamba-based vision encoders to enhance performance in complex power grid environments.

---

## 🌟 Key Features
- **Modern Architecture**: Built on the latest **YOLOv11** for superior speed and accuracy.
- **Multiple Model Variants**:
  - `MSCB (Multi-Scale Context Block)`: Enhanced spatial feature extraction.
  - `WTConv (Wavelet Transform Convolution)`: Improved multi-resolution representation.
  - `Mamba-Vision`: Integration of State Space Models (SSM) for long-range dependency modeling.
- **Thesis-Ready Visualization**: Custom scripts to generate high-quality confusion matrices, PR curves, and loss charts directly for academic publications.
- **Extensible Framework**: Easy to plug in new modules and compare experimental results.

---

## 📂 Project Structure
```text
.
├── train_official.py          # Baseline YOLOv11 training script
├── train_C3k2_MSCB1.py       # Variant 1: C3k2 with MSCB integration
├── train_C3k2_MSCB2.py       # Variant 2: Optimized MSCB structure
├── train-WTConv.py           # Training with Wavelet Transform Convolution
├── train_mamba.py            # Training with Vision Mamba (SSM)
├── visualize_thesis_charts.py # Visualization tool for thesis-grade charts
├── yolo11n.pt                # Official pre-trained weights
├── runs/                     # Training outputs (results, weights, logs)
└── data/                     # (User-defined) Dataset folder (images/labels)
```

## 🛠️ Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/Drop0827/Insulator_System_m.git
   cd Insulator_System_m
   ```

2. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   # Or install core packages directly:
   pip install ultralytics torch torchvision matplotlib seaborn
   ```



## 🚀 Usage

### 1. Training

To start training a specific variant (e.g., the MSCB1 model), run:

```bash
python train_C3k2_MSCB1.py
```

*Note: Ensure your dataset path is correctly configured in your .yaml file.*

### 2. Validation & Inference

Results will be automatically saved in the runs/train/ directory, including:

- weights/best.pt: The best performing model.
- confusion_matrix.png: Model performance per class.
- results.png: Training loss and mAP curves.

### 3. Visualization for Thesis

To generate comparative charts for your paper:

```bash
python visualize_thesis_charts.py
```

## 🤝 Contributing

Contributions are welcome! Feel free to open an **Issue** or submit a **Pull Request** if you have improved models or better visualization scripts.

## 📜 License


This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/url?sa=E&q=LICENSE) file for details.
