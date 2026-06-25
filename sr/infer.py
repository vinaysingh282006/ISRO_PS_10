import os
import cv2
import torch
import rasterio
import numpy as np

from model import RCANLite

# =====================================================
# CONFIG
# =====================================================

#INPUT_FILE = r"sample\tir_200m.npy"
INPUT_FILE = r"sample\tir_200m.npy"
MODEL_FILE = r"checkpoints\best.pth"

OUTPUT_NPY = r"sample\prediction.npy"
OUTPUT_TIF = r"sample\prediction.tif"
OUTPUT_PNG = r"sample\prediction.png"

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("Device:", device)
print("=" * 50)

# =====================================================
# MODEL
# =====================================================

model = RCANLite().to(device)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "model" in checkpoint:
        model.load_state_dict(
            checkpoint["model"]
        )

    else:
        model.load_state_dict(
            checkpoint
        )

else:
    model.load_state_dict(
        checkpoint
    )

model.eval()

# =====================================================
# LOAD INPUT
# =====================================================

ext = os.path.splitext(
    INPUT_FILE
)[1].lower()

if ext == ".npy":

    img = np.load(
        INPUT_FILE
    ).astype(np.float32)

elif ext in [".tif", ".tiff"]:

    with rasterio.open(INPUT_FILE) as src:

        img = src.read(1).astype(
            np.float32
        )

else:

    raise ValueError(
        f"Unsupported file type: {ext}"
    )

# normalize same way as training

if img.max() > 1:
    img /= 65535.0

print("\nInput Statistics")
print("----------------")
print("Shape :", img.shape)
print("Min   :", img.min())
print("Max   :", img.max())
print("Mean  :", img.mean())

# H,W -> 1,H,W

if img.ndim == 2:

    img = np.expand_dims(
        img,
        axis=0
    )

img = torch.from_numpy(
    img
).float()

# 1,H,W -> 1,1,H,W

if img.ndim == 3:

    img = img.unsqueeze(0)

img = img.to(device)

# =====================================================
# INFERENCE
# =====================================================

with torch.no_grad():

    pred = model(
        img
    )

print("\nPrediction Statistics")
print("----------------------")
#print("Shape :", pred.shape)
#print("Min   :", pred.min().item())
#print("Max   :", pred.max().item())
#print("Mean  :", pred.mean().item())


print("Pred Min:", pred.min())
print("Pred Max:", pred.max())
print("Pred Mean:", pred.mean())
print("Pred Std :", pred.std())
print("Input Std :", img.std().item())
print("Pred Std  :", pred.std().item())
# =====================================================
# TO NUMPY
# =====================================================

pred = (
    pred.squeeze()
    .cpu()
    .numpy()
)

pred = np.clip(
    pred,
    0,
    1
)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
plt.hist(
    pred.flatten(),
    bins=256
)
plt.title("Prediction Histogram")
plt.show()
# =====================================================
# SAVE RAW NPY
# =====================================================

np.save(
    OUTPUT_NPY,
    pred
)

print("\nSaved:", OUTPUT_NPY)

# =====================================================
# SAVE TIFF
# =====================================================

pred_uint16 = (
    pred * 65535.0
).astype(np.uint16)

with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    height=pred_uint16.shape[0],
    width=pred_uint16.shape[1],
    count=1,
    dtype=np.uint16
) as dst:

    dst.write(
        pred_uint16,
        1
    )

print("Saved:", OUTPUT_TIF)
# =====================================================
# SAVE PNG (AGGRESSIVE VISUALIZATION)
# =====================================================

pred_vis = pred.copy().astype(np.float32)

# -----------------------------------
# 1. Percentile Stretch
# -----------------------------------

p1 = np.percentile(pred_vis, 1)
p99 = np.percentile(pred_vis, 99)

pred_vis = np.clip(
    pred_vis,
    p1,
    p99
)

pred_vis = (
    pred_vis - p1
) / (
    p99 - p1 + 1e-8
)

# -----------------------------------
# 2. Gamma Boost
# -----------------------------------

gamma = 0.45

pred_vis = np.power(
    pred_vis,
    gamma
)

# -----------------------------------
# 3. Convert to uint8
# -----------------------------------

pred_vis = (
    pred_vis * 255
).astype(np.uint8)

# -----------------------------------
# 4. CLAHE
# -----------------------------------

clahe = cv2.createCLAHE(
    clipLimit=6.0,
    tileGridSize=(8, 8)
)

pred_vis = clahe.apply(
    pred_vis
)

# -----------------------------------
# 5. Unsharp Mask
# -----------------------------------

blur = cv2.GaussianBlur(
    pred_vis,
    (0, 0),
    2.0
)

pred_vis = cv2.addWeighted(
    pred_vis,
    1.8,
    blur,
    -0.8,
    0
)

# -----------------------------------
# 6. Optional Contrast Boost
# -----------------------------------

pred_vis = cv2.convertScaleAbs(
    pred_vis,
    alpha=1.35,
    beta=0
)

# -----------------------------------
# SAVE
# -----------------------------------

cv2.imwrite(
    OUTPUT_PNG,
    pred_vis
)

print("Saved:", OUTPUT_PNG)



# =====================================================
# DONE
# =====================================================

print("\nFinal Output Shape:", pred.shape)
print("=" * 50)