import torch
import math
from pytorch_msssim import ssim


def calculate_psnr(pred, target):

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return 100

    return (
        20
        * math.log10(1.0 / math.sqrt(mse.item()))
    )


def calculate_ssim(pred, target):

    return ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True
    ).item()