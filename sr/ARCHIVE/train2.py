

import torch.optim as optim
import torch.nn as nn
import torch
import random
import math
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.amp import autocast
from torch.amp import GradScaler
from pytorch_msssim import ms_ssim
from dataset import TIRSuperResolutionDataset
from model import RCANLite

import model

print("\nMODEL PATH:")
print(model.__file__)

test_model = RCANLite()

print(
    "TRAIN PARAMS:",
    sum(
        p.numel()
        for p in test_model.parameters()
    )
)


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" peice of shit just caushing problams


# ==========================
# CONFIG
# ==========================

DATASET_PATH = r"C:\Users\Vinay Singh\Desktop\ISRO_PS_10\output\patches"

BATCH_SIZE = 4
EPOCHS = 100

LR = 1e-4

NUM_WORKERS = 0

CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42


# ==========================
# REPRODUCIBILITY
# ==========================

random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ==========================
# DEVICE
# ==========================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA NOT FOUND!"
    )

device = torch.device("cuda")
print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(0))
# RTX 4060 optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print()
print("=" * 50)
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)
print()

# ==========================
# DATASET
# ==========================

dataset = TIRSuperResolutionDataset(
    DATASET_PATH
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset)
    - train_size
)

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False
)


# ==========================
# MODEL
# ==========================

model = RCANLite().to(device)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

# ==========================
# LOSS
# ==========================
# ==========================
# LOSS
# ==========================
l1_loss = nn.L1Loss()
r"""



def combined_loss(pred, target):

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    l1 = l1_loss(pred, target)

    ssim = 1 - ms_ssim(
        pred.float(),
        target.float(),
        data_range=1.0
    )

    return (
        0.85 * l1 +
        0.15 * ssim
    ) """



def combined_loss(pred, target):

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        return l1_loss(pred, target)
# ==========================
# OPTIMIZER
# ==========================


optimizer = optim.AdamW(
    model.parameters(),
    lr=LR
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = GradScaler("cuda")


# ==========================
# PSNR
# ==========================

def psnr(pred, target):

    pred = torch.clamp(
        pred,
        0.0,
        1.0
    )

    target = torch.clamp(
        target,
        0.0,
        1.0
    )

    mse = torch.mean(
        (pred - target) ** 2
    )

    mse = torch.clamp(
        mse,
        min=1e-10
    )

    return (
        20 * torch.log10(
            torch.tensor(
                1.0,
                device=pred.device
            ) / torch.sqrt(mse)
        )
    ).item()
# ==========================
# CHECKPOINT
# ==========================


best_reward = -999999

start_epoch = 1

last_ckpt = os.path.join(
    CHECKPOINT_DIR,
    "last.pth"
)

if os.path.exists(last_ckpt):

    print(
        "Resuming training..."
    )

    ckpt = torch.load(
        last_ckpt
    )

    model.load_state_dict(
        ckpt["model"]
    )

    optimizer.load_state_dict(
        ckpt["optimizer"]
    )

    start_epoch = (
        ckpt["epoch"]
        + 1
    )

    best_reward = (
        ckpt["best_reward"]
    )


# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    model.train()

    train_loss = 0

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch}"
    )

    for lr_img, hr_img in loop:

        lr_img = lr_img.to(
            device,
            non_blocking=True
        )

        hr_img = hr_img.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast("cuda"):
            pred = model(lr_img)
            loss = combined_loss(
                pred,
                hr_img
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )
        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        loop.set_postfix(
            loss=loss.item()
        )

    scheduler.step()

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    val_loss = 0
    val_psnr = 0

    with torch.no_grad():

        for lr_img, hr_img in val_loader:

            lr_img = lr_img.to(
                device,
                non_blocking=True
            )

            hr_img = hr_img.to(
                device,
                non_blocking=True
            )

            pred = model(
                lr_img
            )

            if torch.isnan(pred).any():
                raise RuntimeError(
                    "NaN detected in model output!"
                )
            loss = combined_loss(
                pred,
                hr_img
            )
            if torch.isnan(loss).any():
                raise RuntimeError(
                    "NaN detected in loss!"
                )

            val_loss += loss.item()

            val_psnr += psnr(
                pred,
                hr_img
            )

    val_loss /= len(
        val_loader
    )

    val_psnr /= len(
        val_loader
    )

    reward = (
        val_psnr
        - 10 * val_loss
    )

    print()
    print(
        f"Epoch {epoch}"
    )
    train_loss /= len(train_loader)

    print(
        f"Train Loss : {train_loss:.4f}"
    )
    print(
        f"Val Loss   : {val_loss:.4f}"
    )
    print(
        f"PSNR       : {val_psnr:.2f}"
    )
    print(
        f"Reward     : {reward:.2f}"
    )
    print()

    # =====================
    # SAVE LAST
    # =====================

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_reward": best_reward
        },
        last_ckpt
    )

    # =====================
    # SAVE BEST
    # =====================

    if reward > best_reward:

        best_reward = reward

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best.pth"
            )
        )

        print(
            "NEW BEST MODEL SAVED"
        )

        # do not deleet this code this works but do not support multiple num worjkers and it is not optimized for speed but it works and it is stable

        # next code was't working we have to use the above code because it is optimized for speed and it is stable and it supports multiple num workers but the below code is
