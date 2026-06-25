import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CHANGE THIS IF NEEDED
# ==========================================

FILE = r"sample\prediction.npy"

# ==========================================

img = np.load(FILE)

print("=" * 50)
print("Shape :", img.shape)
print("Min   :", img.min())
print("Max   :", img.max())
print("Mean  :", img.mean())
print("Std   :", img.std())
print("=" * 50)

# Contrast stretch
p2 = np.percentile(img, 2)
p98 = np.percentile(img, 98)

img_vis = np.clip(img, p2, p98)

plt.figure(figsize=(10, 10))
plt.imshow(img_vis, cmap="gray")
plt.colorbar()
plt.title(FILE)
plt.axis("off")

plt.show()