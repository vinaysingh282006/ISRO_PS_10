import cv2
import numpy as np

lr = np.load("sample/tir_200m.npy")[0]

bicubic = cv2.resize(
    lr,
    (lr.shape[1]*2, lr.shape[0]*2),
    interpolation=cv2.INTER_CUBIC
)

np.save("sample/bicubic.npy", bicubic)