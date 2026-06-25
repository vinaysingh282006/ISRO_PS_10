# 🚀 Bhartiya Antriksh Hackathon (BAH) 2026

## Problem Statement: Infrared Image Colorization and Enhancement for Improved Object Interpretation

Welcome to the **Bhartiya Antriksh Hackathon 2026**! This repository provides a baseline implementation and technical guidelines for the challenge of transforming raw Thermal Infrared (TIR) satellite imagery into interpretable, colorized visual representations.

---

## The Challenge

Thermal Infrared (TIR) data is invaluable for monitoring wildfires, urban heat islands, and volcanic activity. However, raw TIR imagery is typically single-band (grayscale) and lacks the intuitive detail of RGB imagery, making object interpretation difficult for human analysts.

**Your Goal:** Develop a computational pipeline and machine learning model that produces two primary outputs:
1. **A Super-Resolved TIR Image**: Increase the spatial resolution of raw TIR imagery to recover critical structural details.
2. **A Colorized TIR Image**: Synthesize realistic colors for the TIR data, using multi-spectral RGB data as a guide.

---

## Data Acquisition

### Data Specifications
On the USGS Earth Explorer site, all Landsat 9 bands (B2, B3, B4, and B10) are registered and provided at a **30m resolution**. However, it is important to note that the original spatial resolution of the TIR band (B10) is **100m**, while the RGB bands (B2, B3, B4) are natively **30m**.

To get started, you will need Landsat 9 imagery.

### Quick Start (Demo Data)
Use the provided bash script to download sample bands into `input/demo_product/`:
```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

### Custom Downloads (Google Earth Engine)
Use `scripts/download.py` to fetch specific bands using GEE:
```bash
python scripts/download.py <product_id> <bands> <start_date> <end_date> <output_path> --ee_project_id <your_project_id>
```

### Custom Downloads (USGS Earth Explorer)
You may also download data directly from [USGS Earth Explorer](https://earthexplorer.usgs.gov/); please ensure it is placed in the `input` directory following the structure below.

### Required Input Directory Structure
To ensure the baseline scripts function correctly, please organize your data as follows:
```
input/
└── <folder_name>/
    ├── <file_prefix>_B10.TIF
    ├── <file_prefix>_B2.TIF
    ├── <file_prefix>_B3.TIF
    └── <file_prefix>_B4.TIF
```
*Note: While `<folder_name>` can be any identifier of your choice, the files inside must end with the specified band suffixes (`_B10.TIF`, `_B2.TIF`, `_B3.TIF`, `_B4.TIF`) to be correctly processed by the pipeline.*

---

## Baseline Implementation: Dataset Generation

This baseline focuses on the most critical part of the pipeline: **creating co-registered training pairs**.

### Dataset Generation Workflow
Run the driver script to generate multi-resolution, spatially aligned patches:
```bash
python driver.py
```

**Pipeline Details:**
1. **Merge**: Optical bands (B2, B3, B4) are merged into a 30m RGB image.
2. **Downscale**: The baseline takes the 30m resampled USGS data and downscales it to create training pairs:
   - **Input**: All bands are processed from their 30m resampled versions.
   - **Rescaling Factors**:
     - RGB (30m) $\xrightarrow{\times 3.33}$ 100m
     - TIR (30m) $\xrightarrow{\times 3.33}$ 100m
     - TIR (30m) $\xrightarrow{\times 6.67}$ 200m
   - **Data Flow**: For the Super-Resolution task, the TIR band is downsampled to 200m as input, with the objective of recovering a 100m output.
3. **Extract Co-registered Patches**:
   - **SR Pair**: 256x256 (200m TIR) $\rightarrow$ 512x512 (100m TIR).
   - **Colorization Pair**: 256x256 (100m TIR) $\rightarrow$ 256x256 (100m RGB).
4. **Save Output**: Both `.npy` (for training) and `.png` (for verification) files are saved in `output/patches/`.
   - ⚠️ **Important**: Do **not** train your models on the `.png` files. `.png` files are intended for visualization purposes only. For training, use the `.npy` files to maintain the original radiometric resolution of the data.

### Technical Alignment
The baseline ensures strict spatial co-registration:
- One pixel in the 200m TIR image corresponds exactly to a 2x2 block in the 100m TIR/RGB images.
- All patches are extracted using the same top-left offset to maintain alignment across resolutions.

---

## Conceptual Workflow

The following diagram illustrates the end-to-end process for dataset generation. While we provide a baseline, these are **suggested approaches**. You are encouraged to explore alternative workflows for better structural accuracy or design entirely new pipelines to achieve the objectives.

```mermaid
graph TD
    A[Start: Raw Data Source] --> B(Download Landsat 9 Bands: B2, B3, B4, B10 - 30m)
    
    B --> C1(Merge B2, B3, B4 into RGB Image - 30m)
    B --> C2(Downscale TIR B10 by 3.33x - 100m)
    B --> C3(Downscale TIR B10 by 6.67x - 200m)
    
    C1 --> D(Downscale RGB by 3.33x - 100m)
    
    D --> E1(Create Image Patches: 100m RGB & 100m TIR)
    C2 --> E2(Create Image Patches: 100m TIR & 200m TIR)
    C3 --> E2
```

---

## Stage 1: Thermal Super-Resolution (Implemented)

This repository contains a complete, research-grade pipeline for Stage 1: Super-Resolution, scaling 200m TIR up to 100m TIR. 

### Architecture Choice: RCAN
For a small dataset (721 samples) and consumer hardware (e.g., RTX 4060 8GB), transformer models (SwinIR/HAT) are not practical due to extreme VRAM requirements and data hunger. 
We implemented an **RCAN (Residual Channel Attention Network)** which fits perfectly within these constraints while providing state-of-the-art detail recovery.
- **Channel Attention**: Upweights feature maps that detect subtle temperature gradients.
- **PixelShuffle**: Artifact-free 2x upsampling.
- **GPU Config**: 15.4M parameters, perfectly sized for 8GB VRAM.

### Folder Structure
- `sr/config.py`: Hyperparameters (batch size, epochs, architecture size)
- `sr/dataset.py`: DataLoader, train/val split, uint16 normalization
- `sr/model.py`: RCAN PyTorch implementation
- `sr/train_sr.py`: Full training loop with AMP (Mixed Precision) and validation
- `sr/infer_sr.py`: Inference script
- `sr/metrics.py`: PSNR, SSIM, SSIM-loss, Bicubic baseline
- `sr/utils.py`: Checkpoint saving, visual comparisons
- `run_sr.bat`: Automated environment setup and training launcher

### How to Run
1. **Check Pipeline**: Run `python sr/check_pipeline.py` to ensure your GPU is used correctly and tensors match expected shapes.
2. **Install and Train**: Run `run_sr.bat` in your terminal. It automatically detects your GPU, installs the specific CUDA-enabled PyTorch build for Python 3.14, and begins training.
3. **Resume Training**: `run_sr.bat --resume`
4. **Inference**: `python sr/infer_sr.py --input output/patches/D-1/sample_000/tir_200m.npy`

*Logs, visual comparisons, and `.pth` checkpoints are saved into `sr/logs`, `sr/eval_outputs`, and `sr/checkpoints` respectively.*

---

## Expected Pipeline and Output Format

Following the dataset generation, you are expected to implement a multi-stage model pipeline.

**Inference Flow:**
For the entire pipeline during inference, the input will be the **200m resolution TIR band (B10)**. The pipeline is expected to produce the two outputs detailed below.

1. **Super-Resolution Stage**: Develop a model to generate high-resolution (100m) TIR images from the low-resolution (200m) inputs.
2. **Colorization Stage**: Pass the resulting high-resolution TIR images into a colorization model to produce synthetic, interpretable RGB representations.

### Mandatory Output Structure
To ensure standardized evaluation, your final output must be organized in the `output/` directory as follows:

```
output/
└── model_outputs/
    ├── tir_superresolved_100m/
    │   └── <product_id>.tif
    └── colorized_tir_100m/
        └── <product_id>.tif
```
*Note: `<product_id>` must exactly match the original input product ID.*

**Band Ordering Requirement:**
For the colorized TIR images, the output TIFF must adhere to the following channel sequence:
- **Layer 1**: Blue
- **Layer 2**: Green
- **Layer 3**: Red

---

## Submission Guidelines

**Required Deliverables:**
1. **Codebase**: A link to your GitHub repository.
2. **Model Weights**: Your trained model weights (e.g., `.pth`, `.h5`).
3. **Technical Report**: A PDF detailing your approach and results.
4. **Sample Results**: A sequence of Raw TIR $\rightarrow$ Super-Resolved TIR $\rightarrow$ Colorized TIR.
