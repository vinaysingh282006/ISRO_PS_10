# ============================================================
# IR SUPER RESOLUTION TRAINER
# Windows Multiprocessing Safe Version
# ============================================================
r"""

# ============================================================
we will get back to it for now train.py works good enough for now

this version is better but i do not get enough time to get it completely working and tested. I will get back to it later.

# ============================================================



import os
import random
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.amp import autocast
from torch.amp import GradScaler

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import TIRSuperResolutionDataset
from model import RCANLite
import model


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = r"C:\Users\Vinay Singh\Desktop\ISRO_ARCHIVE_OUTPUT\patches"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 2

EPOCHS = 100

LR = 1e-4

NUM_WORKERS = 0

SEED = 42

RESUME = False

BEST_MODEL_NAME = "best.pth"

LAST_MODEL_NAME = "last.pth"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    try:

        torch.set_float32_matmul_precision("high")

    except:

        pass


# ============================================================
# LOSS
# ============================================================

l1_loss = nn.L1Loss()


def combined_loss(pred, target):

    return l1_loss(
        pred,
        target
    )


# ============================================================
# PSNR
# ============================================================

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

        20

        *

        torch.log10(

            torch.tensor(
                1.0,
                device=pred.device
            )

            /

            torch.sqrt(mse)

        )

    ).item()


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA NOT FOUND!"
        )

    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print()

    print("=" * 50)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print("=" * 50)

    print()

    # ========================================================
    # DATASET
    # ========================================================

    dataset = TIRSuperResolutionDataset(
        DATASET_PATH
    )

    print(
        "Dataset Size:",
        len(dataset)
    )

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_ds, val_ds = random_split(

        dataset,

        [train_size, val_size],

        generator=generator

    )

    train_loader = DataLoader(

        train_ds,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=(NUM_WORKERS > 0),

        prefetch_factor=2 if NUM_WORKERS > 0 else None

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        persistent_workers=(NUM_WORKERS > 0),

        prefetch_factor=2 if NUM_WORKERS > 0 else None

    )

    # ========================================================
    # MODEL
    # ========================================================

    model = RCANLite().to(device)

    print()

    print(

        "Parameters:",

        sum(

            p.numel()

            for p in model.parameters()

        )

    )

    print()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LR

    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS,

        eta_min=1e-6

    )

    scaler = GradScaler("cuda")

    # ========================================================
    # CHECKPOINT
    # ========================================================

    best_reward = -float("inf")

    start_epoch = 1

    last_ckpt = os.path.join(

        CHECKPOINT_DIR,

        LAST_MODEL_NAME

    )

    best_ckpt = os.path.join(

        CHECKPOINT_DIR,

        BEST_MODEL_NAME

    )

    if RESUME and os.path.exists(last_ckpt):

        print("Resuming training...")

        ckpt = torch.load(

            last_ckpt,

            map_location=device

        )

        model.load_state_dict(

            ckpt["model"]

        )

        optimizer.load_state_dict(

            ckpt["optimizer"]

        )

        start_epoch = ckpt["epoch"] + 1

        best_reward = ckpt["best_reward"]

    # ========================================================
    # TRAIN LOOP STARTS HERE
    # ========================================================

    # ========================================================
# TRAIN LOOP
# ========================================================
    for epoch in range(start_epoch, EPOCHS + 1):

        # -----------------------------
        # TRAIN
        # -----------------------------

        model.train()

    train_loss = 0.0

    loop = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}",
        leave=True
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

        if torch.isnan(loss):

            print("\nNaN LOSS DETECTED\n")

            print(
                "Prediction:",
                pred.min().item(),
                pred.max().item()
            )

            print(
                "Target:",
                hr_img.min().item(),
                hr_img.max().item()
            )

            raise RuntimeError(
                "Loss became NaN."
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
            loss=f"{loss.item():.5f}"
        )

    scheduler.step()

    train_loss /= len(train_loader)

    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

val_loss = 0.0
val_psnr = 0.0

with torch.no_grad():

    for lr_img, hr_img in tqdm(
        val_loader,
        desc="Validation",
        leave=False
    ):

        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        # THIS WAS MISSING
        pred = model(lr_img)

        if torch.isnan(pred).any():
            raise RuntimeError("NaN detected in prediction.")

        loss = combined_loss(pred, hr_img)

        val_loss += loss.item()
        val_psnr += psnr(pred, hr_img)

val_loss /= len(val_loader)
val_psnr /= len(val_loader)

reward = val_psnr - 10.0 * val_loss

print()
print("=" * 60)
print(f"Epoch {epoch}/{EPOCHS}")
print(f"Train Loss : {train_loss:.6f}")
print(f"Val Loss   : {val_loss:.6f}")
print(f"PSNR       : {val_psnr:.2f} dB")
print(f"Reward     : {reward:.3f}")
print("=" * 60)
print()

        # =======================================================
    # SAVE LAST CHECKPOINT
    # =======================================================

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_reward": best_reward,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "reward": reward,
        },
        last_ckpt,
    )

    # =======================================================
    # SAVE BEST MODEL
    # =======================================================

    if reward > best_reward:

        best_reward = reward

        torch.save(
            model.state_dict(),
            best_ckpt
        )

        print("\n★★★★★ NEW BEST MODEL SAVED ★★★★★\n")

    # =======================================================
    # FREE GPU CACHE
    # =======================================================

    torch.cuda.empty_cache()


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":

    main()

    

    """