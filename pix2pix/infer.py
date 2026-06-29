
import os
import time
import numpy as np

import torch
import rasterio
import matplotlib.pyplot as plt
import cv2
from model import build_generator

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = r"checkpoints/generator_best.pth"

INPUT_FILE = r"samples/tir_100m_512.npy"
# INPUT_FILE = r"samples/D-115_B10.tiff"

OUTPUT_DIR = "samples"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print("=" * 60)

print("Device :", DEVICE)

print("=" * 60)

# ==========================================================
# LOAD MODEL
# ==========================================================

generator = build_generator().to(
    DEVICE
)

checkpoint = torch.load(

    MODEL_PATH,

    map_location=DEVICE

)

generator.load_state_dict(
    checkpoint
)

generator.eval()

print()

print("Generator Loaded Successfully")

print()

# ==========================================================
# LOAD THERMAL IMAGE
# ==========================================================

thermal = np.load(
    INPUT_FILE
).astype(np.float32)

print("=" * 60)

print("Input Statistics")

print("=" * 60)

print("Shape :", thermal.shape)

print("Min   :", thermal.min())

print("Max   :", thermal.max())

print("Mean  :", thermal.mean())

# ==========================================================
# NORMALIZE
# ==========================================================

thermal /= 65535.0

thermal = torch.from_numpy(
    thermal
)

thermal = thermal.unsqueeze(0)

thermal = thermal.to(
    DEVICE
)


# ==========================================================
# ISRO STYLE VISUALIZATION
# ==========================================================

def percentile_stretch(img):

    img = img.astype(np.float32)

    out = np.zeros_like(img)

    for c in range(3):

        low = np.percentile(img[..., c], 1)

        high = np.percentile(img[..., c], 99)

        out[..., c] = np.clip(

            (img[..., c]-low)/(high-low+1e-6),

            0,

            1

        )

    return out


# ==========================================================
# COLOR REFINEMENT
# ==========================================================

def enhance_rgb(rgb):

    rgb = np.clip(rgb, 0, 1)

    # Gamma correction
    gamma = 0.85
    rgb = np.power(rgb, gamma)

    # Contrast
    rgb = np.clip((rgb - 0.5) * 1.35 + 0.5, 0, 1)

    # Saturation
    hsv = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.4,
        0,
        255
    )

    rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    ).astype(np.float32) / 255.0

    # CLAHE on luminance
    lab = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2LAB
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    rgb = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    ).astype(np.float32) / 255.0

    # Sharpen
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)

    rgb = cv2.addWeighted(
        rgb,
        1.5,
        blur,
        -0.5,
        0
    )

    return np.clip(rgb, 0, 1)


# ==========================================================
# INFERENCE
# ==========================================================
start = time.time()

with torch.no_grad():

    with torch.autocast(

        device_type="cuda",

        dtype=torch.float16,

        enabled=(DEVICE.type == "cuda")

    ):

        prediction = generator(
            thermal
        )

end = time.time()

prediction = prediction.squeeze(0)

prediction = prediction.float().cpu().numpy()

print()

print("=" * 60)

print("Prediction")

print("=" * 60)

print("Shape :", prediction.shape)

print("Min   :", prediction.min())

print("Max   :", prediction.max())

print("Mean  :", prediction.mean())

print("Std   :", prediction.std())

print()

print(

    f"Inference Time : {end-start:.3f} seconds"

)

print("=" * 60)

print()

print("Saving outputs...")


# ==========================================================
# POST PROCESS
# ==========================================================

prediction = np.clip(prediction, 0.0, 1.0)

# Network output (H,W,C)
rgb = np.transpose(prediction, (1, 2, 0))

# -------- ISRO style visualization --------

rgb_vis = percentile_stretch(rgb)

# Very light denoising (removes speckle)
rgb_vis = cv2.bilateralFilter(
    (rgb_vis * 255).astype(np.uint8),
    5,
    25,
    7
).astype(np.float32) / 255.0

# ==========================================================
# SAVE NPY (RAW FLOAT)
# ==========================================================

npy_path = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction.npy"
)

np.save(
    npy_path,
    prediction
)

print("Saved :", npy_path)

# ==========================================================
# SAVE TIFF (RAW MODEL OUTPUT)
# ==========================================================

tif_path = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction.tif"
)

rgb16 = (prediction * 65535).astype(np.uint16)

with rasterio.open(

    tif_path,

    "w",

    driver="GTiff",

    height=prediction.shape[1],

    width=prediction.shape[2],

    count=3,

    dtype=np.uint16,

    photometric="RGB",

    interleave="pixel"

) as dst:

    dst.write(rgb16)

print("Saved :", tif_path)

# ==========================================================
# SAVE PNG (VISUALIZATION)
# ==========================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction.png"
)

plt.figure(figsize=(8, 8))

plt.imshow(rgb_vis)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    png_path,
    dpi=200,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("Saved :", png_path)

# ==========================================================
# SAVE HISTOGRAM
# ==========================================================

hist_path = os.path.join(
    OUTPUT_DIR,
    "rgb_histogram.png"
)

plt.figure(figsize=(8, 5))

colors = ["red", "green", "blue"]

for i in range(3):

    plt.hist(
        prediction[i].ravel(),
        bins=256,
        alpha=0.5,
        color=colors[i]
    )

plt.tight_layout()

plt.savefig(hist_path)

plt.close()

print("Saved :", hist_path)

# ==========================================================

print()
print("="*60)
print("Inference Finished Successfully")
print("="*60)

torch.cuda.empty_cache()


r"""

import os
import time

import cv2
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

from model import build_generator

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = r"checkpoints/generator_best.pth"

INPUT_FILE = r"samples/tir_100m_512.npy"

OUTPUT_DIR = "samples"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", DEVICE)
print("=" * 60)

# ==========================================================
# LOAD MODEL
# ==========================================================

generator = build_generator().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

generator.load_state_dict(checkpoint)

generator.eval()

print()
print("Generator Loaded Successfully")
print()

# ==========================================================
# LOAD INPUT
# ==========================================================

thermal = np.load(
    INPUT_FILE
).astype(np.float32)

print("=" * 60)
print("Input Statistics")
print("=" * 60)

print("Shape :", thermal.shape)
print("Min   :", thermal.min())
print("Max   :", thermal.max())
print("Mean  :", thermal.mean())

thermal /= 65535.0

thermal = torch.from_numpy(
    thermal
).unsqueeze(0).to(DEVICE)

# ==========================================================
# VISUALIZATION
# ==========================================================


def percentile_stretch(
    image,
    low=2,
    high=98
):

    image = image.astype(np.float32)

    result = np.zeros_like(image)

    for c in range(3):

        p_low = np.percentile(
            image[..., c],
            low
        )

        p_high = np.percentile(
            image[..., c],
            high
        )

        result[..., c] = np.clip(
            (image[..., c] - p_low)
            /
            (p_high - p_low + 1e-6),
            0,
            1
        )

    return result

# ==========================================================
# EDGE PRESERVING DENOISE
# ==========================================================


def clean_prediction(rgb):

    rgb8 = (
        np.clip(rgb, 0, 1) * 255
    ).astype(np.uint8)

    # removes checkerboard artifacts
    rgb8 = cv2.fastNlMeansDenoisingColored(
        rgb8,
        None,
        4,
        4,
        7,
        21
    )

    rgb = rgb8.astype(np.float32) / 255.0

    # very light unsharp mask
    blur = cv2.GaussianBlur(
        rgb,
        (0, 0),
        0.8
    )

    rgb = cv2.addWeighted(
        rgb,
        1.15,
        blur,
        -0.15,
        0
    )

    return np.clip(rgb, 0, 1)


# ==========================================================
# INFERENCE (TTA)
# ==========================================================
print()
print("Running inference...")
print()

start = time.time()

with torch.no_grad():

    # -----------------------------
    # Original
    # -----------------------------
    pred0 = generator(thermal)

    # -----------------------------
    # Horizontal Flip
    # -----------------------------
    thermal_h = torch.flip(
        thermal,
        dims=[3]
    )

    pred_h = generator(
        thermal_h
    )

    pred_h = torch.flip(
        pred_h,
        dims=[3]
    )

    # -----------------------------
    # Vertical Flip
    # -----------------------------
    thermal_v = torch.flip(
        thermal,
        dims=[2]
    )

    pred_v = generator(
        thermal_v
    )

    pred_v = torch.flip(
        pred_v,
        dims=[2]
    )

    # -----------------------------
    # Horizontal + Vertical
    # -----------------------------
    thermal_hv = torch.flip(
        thermal,
        dims=[2, 3]
    )

    pred_hv = generator(
        thermal_hv
    )

    pred_hv = torch.flip(
        pred_hv,
        dims=[2, 3]
    )

    # -----------------------------
    # Average Prediction
    # -----------------------------
    prediction = (

        pred0 +

        pred_h +

        pred_v +

        pred_hv

    ) / 4.0

end = time.time()

prediction = prediction.squeeze(0)

prediction = prediction.float().cpu().numpy()

prediction = np.clip(
    prediction,
    0,
    1
)

print("=" * 60)
print("Prediction Statistics")
print("=" * 60)

print("Shape :", prediction.shape)
print("Min   :", prediction.min())
print("Max   :", prediction.max())
print("Mean  :", prediction.mean())
print("Std   :", prediction.std())

print()
print(
    f"Inference Time : {end-start:.3f} seconds"
)

print("=" * 60)

# ==========================================================
# PREPARE RGB
# ==========================================================

rgb = np.transpose(
    prediction,
    (1, 2, 0)
)

# Remove tiny GAN artifacts
rgb = clean_prediction(
    rgb
)

# Visualization only
rgb_vis = percentile_stretch(
    rgb
)

print()
print("Preparing output images...")
print()


# ==========================================================
# SAVE NPY
# ==========================================================

npy_path = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction.npy"
)

np.save(
    npy_path,
    prediction
)

print("Saved :", npy_path)

# ==========================================================
# SAVE TIFF (16-bit RGB)
# ==========================================================

tif_path = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction.tif"
)

rgb16 = (
    np.clip(rgb, 0, 1) * 65535
).astype(np.uint16)

with rasterio.open(

    tif_path,

    "w",

    driver="GTiff",

    height=rgb16.shape[0],

    width=rgb16.shape[1],

    count=3,

    dtype=np.uint16,

    photometric="RGB",

    interleave="pixel"

) as dst:

    dst.write(
        rgb16[:, :, 0],
        1
    )

    dst.write(
        rgb16[:, :, 1],
        2
    )

    dst.write(
        rgb16[:, :, 2],
        3
    )

print("Saved :", tif_path)

# ==========================================================
# SAVE PNG
# ==========================================================

png_path = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction.png"
)

plt.figure(
    figsize=(8, 8)
)

plt.imshow(rgb_vis)

plt.axis("off")

plt.tight_layout()

plt.savefig(

    png_path,

    dpi=250,

    bbox_inches="tight",

    pad_inches=0

)

plt.close()

print("Saved :", png_path)

# ==========================================================
# SAVE RAW PNG
# ==========================================================

raw_png = os.path.join(
    OUTPUT_DIR,
    "rgb_prediction_raw.png"
)

plt.figure(
    figsize=(8, 8)
)

plt.imshow(rgb)

plt.axis("off")

plt.tight_layout()

plt.savefig(

    raw_png,

    dpi=250,

    bbox_inches="tight",

    pad_inches=0

)

plt.close()

print("Saved :", raw_png)

# ==========================================================
# SAVE HISTOGRAM
# ==========================================================

hist_path = os.path.join(
    OUTPUT_DIR,
    "rgb_histogram.png"
)

plt.figure(
    figsize=(10, 5)
)

colors = [
    "red",
    "green",
    "blue"
]

for i in range(3):

    plt.hist(

        prediction[i].ravel(),

        bins=256,

        color=colors[i],

        alpha=0.45,

        label=colors[i]

    )

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    hist_path
)

plt.close()

print("Saved :", hist_path)

# ==========================================================
# FINISHED
# ==========================================================

print()

print("="*60)
print("Inference Finished Successfully")
print("="*60)

print()

print("Generated Files")
print("----------------")
print(npy_path)
print(tif_path)
print(png_path)
print(raw_png)
print(hist_path)

torch.cuda.empty_cache()
"""
