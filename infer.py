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
#INPUT_FILE = r"samples/D-115_B10.tiff"

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
# CONVERT TO UINT16
# ==========================================================

prediction = np.clip(
    prediction,
    0,
    1
)

rgb = np.transpose(
    prediction,
    (1, 2, 0)
)

rgb = enhance_rgb(rgb)

rgb_vis = percentile_stretch(rgb)

prediction_uint16 = (
    rgb.transpose(2, 0, 1) * 65535
).astype(np.uint16)

# ==========================================================
# SAVE NPY
# ==========================================================

npy_path = os.path.join(

    OUTPUT_DIR,

    "rgb_prediction.npy"

)

np.save(

    npy_path,

    prediction_uint16

)

print("Saved :", npy_path)

# ==========================================================
# SAVE TIFF
# ==========================================================
prediction_uint16 = (
    prediction * 65535
).astype(np.uint16)

# ==========================================================
# SAVE RAW PNG
# ==========================================================
rgb = np.transpose(
    prediction,
    (1,2,0)
)

rgb = percentile_stretch(rgb)
# ==========================================================
# SAVE NORMALIZED PNG
# ==========================================================

png_path = os.path.join(

    OUTPUT_DIR,

    "rgb_prediction.png"

)


rgb = np.transpose(
    prediction,
    (1, 2, 0)
)

rgb = enhance_rgb(rgb)

rgb_vis = percentile_stretch(rgb)

plt.figure(

    figsize=(8, 8)

)

plt.imshow(rgb_vis)

plt.axis("off")

plt.tight_layout()

plt.savefig(

    png_path,

    dpi=200,

    bbox_inches="tight"

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

plt.figure(

    figsize=(8, 5)

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

        alpha=0.5,

        color=colors[i],

        label=f"Channel {i}"

    )

plt.legend()

plt.xlabel("Pixel Value")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(

    hist_path,

    dpi=200

)

plt.close()

print("Saved :", hist_path)

# ==========================================================
# FINISHED
# ==========================================================

print()

print("=" * 60)

print("Inference Finished Successfully")

print()

print("RGB Shape :", prediction.shape)

print()

print("Files Generated")

print("----------------")

print(npy_path)

#print(tif_path)

print(png_path)

#print(raw_png)

print(hist_path)

print("=" * 60)

torch.cuda.empty_cache()
