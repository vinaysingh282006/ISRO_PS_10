import numpy as np
import matplotlib.pyplot as plt

hr = np.load(
    r"C:\Users\Vinay Singh\Desktop\ISRO_PS_10\output\patches\D-9\sample_003\rgb_100m_512.npy"
)

print("Min :", hr.min())
print("Max :", hr.max())
print("Mean:", hr.mean())
print("Std :", hr.std())

plt.hist(
    hr.flatten(),
    bins=256
)

plt.title("Ground Truth Histogram")
plt.show()