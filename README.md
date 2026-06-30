<div align="center">
  <h1>🚀 ISRO Problem Statement 10 (ISRO_PS_10)</h1>
  <p><strong>Super-Resolution & Image-to-Image Translation of Thermal Infrared (TIR) Imagery to RGB</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  </p>
</div>

---

## 📖 Overview

This repository contains deep learning models and scripts developed for **ISRO Problem Statement 10**. The project focuses on processing and enhancing satellite and thermal imagery using advanced neural networks.

The solution is divided into two primary pipelines:
1. **`pix2pix/`**: 🎨 **Image-to-Image Translation** (Translates TIR images to RGB using a U-Net Generator & PatchGAN Discriminator).
2. **`sr/`**: 🔍 **Super-Resolution** (Upscales and enhances thermal image resolution using the RCANLite architecture).

---

## 📂 Folder Structure

```text
ISRO_PS_10/
├── .gitignore
├── infer.py                # Main inference pipeline
├── README.md               # Project documentation
├── pix2pix/                # Image-to-Image Translation Module
│   ├── checkpoints/        # Saved model weights
│   ├── samples/            # Generated outputs
│   ├── dataset.py          # Data loaders
│   ├── infer.py            # Pix2Pix specific inference
│   ├── losses.py           # Custom GAN losses
│   ├── model.py            # U-Net & PatchGAN architectures
│   ├── train.py            # Training script
│   └── utils.py            # Utility functions
└── sr/                     # Super-Resolution Module
    ├── checkpoints/        # Saved model weights
    ├── sample/             # Generated outputs
    ├── Bicubic.py          # Baseline interpolation
    ├── dataset.py          # SR data loaders
    ├── gpu_memory_test.py  # Hardware testing
    ├── gpu_test.py         # Hardware testing
    ├── graph.py            # Plotting utilities
    ├── infer.py            # SR specific inference
    ├── metrics.py          # PSNR & SSIM calculations
    ├── model.py            # RCANLite architecture
    ├── see.py              # Visualization tools
    ├── test_model.py       # Model debugging
    ├── train.py            # Main SR training script
    ├── train2.py           # Alternative training script
    └── train3.py           # Alternative training script
```

---

## 🛠️ Detailed Component Breakdown

### 🎯 1. Main Pipeline (Root)
| File | Description |
|------|-------------|
| ⚙️ **`infer.py`** | The primary inference script. Loads a pre-trained Pix2Pix generator, processes a thermal `.npy` array, normalizes it, applies color enhancement and ISRO-style percentile stretching. Outputs predicted RGB `.png`, `.npy`, and color histograms. |

### 🎨 2. Pix2Pix (Image-to-Image Translation)
| File | Description |
|------|-------------|
| 🧠 **`model.py`** | Neural network architectures including `Generator` (U-Net with Down/Up Blocks) and `Discriminator` (PatchGAN). |
| 🚀 **`train.py`** | Training loop implementing mixed-precision (AMP), AdamW optimizer, and custom loss combinations. |
| 📉 **`losses.py`** | Custom GAN losses: `GeneratorLoss` (L1, perceptual, edge weights) and `DiscriminatorLoss`. |
| 📊 **`dataset.py`**| `RGBDataset` class for loading and preprocessing image patches. |
| 🛠️ **`utils.py`** | Contains utility and helper functions for the Pix2Pix pipeline. |
| 🏃 **`infer.py`** | A localized inference script specific to the Pix2Pix training outputs. |


### 🔍 3. SR (Super-Resolution)
| File | Description |
|------|-------------|
| 🧠 **`model.py`** | Defines `RCANLite` (Residual Channel Attention Network) for efficient image upsampling. |
| 🚀 **`train.py`** | Primary Super-Resolution training script. *(Includes `train2.py`, `train3.py` for experiments).* |
| 📊 **`dataset.py`**| `TIRSuperResolutionDataset` class for loading low-resolution and high-resolution image pairs. |
| 📏 **`metrics.py`**| Evaluates model performance using **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index Measure). |
| 🖼️ **`Bicubic.py`**| Standard bicubic interpolation baseline for model benchmarking. |
| 🧪 **`test_model.py`** | Script for quickly testing model instantiation and basic forward passes. |

---

## 💻 Prerequisites & Installation

Ensure you have the following dependencies installed in your Python environment:

```bash
pip install torch torchvision torchmetrics numpy opencv-python matplotlib rasterio tqdm
```
*Note: A CUDA-enabled GPU is highly recommended for training and fast inference.*

---

## 🚀 Quick Start Guide

### 1️⃣ Training Pix2Pix
Navigate to the `pix2pix/` directory and execute the training script:
```bash
cd pix2pix
python train.py
```
*(Ensure dataset paths are properly configured inside `train.py` before running).*

### 2️⃣ Training Super-Resolution (SR)
Navigate to the `sr/` directory and execute:
```bash
cd sr
python train.py
```

### 3️⃣ Running Inference
From the root directory, ensure you have a trained checkpoint (e.g., `checkpoints/generator_best.pth`) and your input thermal file.
```bash
python infer.py
```
Outputs (RGB predictions, histograms) will automatically be saved to the `samples/` directory.

---
<div align="center">
  <i>Developed for Bhartiya Antariksh Hackathon 2026</i>
</div>
