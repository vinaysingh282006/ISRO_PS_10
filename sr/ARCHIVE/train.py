import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

from dataset import TIRSuperResolutionDataset
from model import RCANLite


# ==========================
# CONFIG
# ==========================

DATASET_PATH = r"C:\Users\Vinay Singh\Desktop\ISRO_PS_10\output\patches"

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
NUM_WORKERS = 0

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SEED = 42
RESUME = False  # set True only if you want to continue from a compatible checkpoint


# ==========================
# REPRODUCIBILITY
# ==========================

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ==========================
# DEVICE
# ==========================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA NOT FOUND!")

device = torch.device("cuda")

print()
print("=" * 50)
print("GPU:", torch.cuda.get_device_name(0))
print("=" * 50)
print()


# ==========================
# DATASET
# ==========================

dataset = TIRSuperResolutionDataset(DATASET_PATH)

if len(dataset) < 2:
    raise RuntimeError("Dataset too small. Check your input folder.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

split_gen = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)


# ==========================
# MODEL
# ==========================

torch.cuda.empty_cache()

model = RCANLite().to(device)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# ==========================
# LOSS
# ==========================

l1_loss = nn.L1Loss()


def combined_loss(pred, target):
    # Keep loss simple and stable for now.
    return l1_loss(pred, target)


# ==========================
# OPTIMIZER
# ==========================

optimizer = optim.AdamW(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

scaler = GradScaler("cuda")


# ==========================
# METRICS
# ==========================

def psnr(pred, target):
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10)

    return (
        20.0
        * torch.log10(
            torch.tensor(1.0, device=pred.device) / torch.sqrt(mse)
        )
    ).item()


# ==========================
# CHECKPOINT
# ==========================

best_reward = -float("inf")
start_epoch = 1

last_ckpt = os.path.join(CHECKPOINT_DIR, "last.pth")
best_ckpt = os.path.join(CHECKPOINT_DIR, "best.pth")

if RESUME and os.path.exists(last_ckpt):
    try:
        print("Resuming training...")

        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_reward = float(ckpt.get("best_reward", -float("inf")))

    except Exception as e:
        print(f"Could not resume from checkpoint: {e}")
        print("Starting fresh training.")


# ==========================
# TRAIN LOOP
# ==========================

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()

    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}")

    for lr_img, hr_img in loop:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pred = model(lr_img)
            loss = combined_loss(pred, hr_img)

        if torch.isnan(loss):
            print("Pred Min:", pred.min().item())
            print("Pred Max:", pred.max().item())
            print("Target Min:", hr_img.min().item())
            print("Target Max:", hr_img.max().item())
            raise RuntimeError("NaN loss detected")

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    # =====================
    # VALIDATION
    # =====================

    model.eval()
    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            pred = model(lr_img)

            if torch.isnan(pred).any():
                raise RuntimeError("NaN detected in model output!")

            loss = combined_loss(pred, hr_img)

            if torch.isnan(loss):
                raise RuntimeError("NaN detected in loss!")

            val_loss += loss.item()
            val_psnr += psnr(pred, hr_img)

    val_loss /= max(1, len(val_loader))
    val_psnr /= max(1, len(val_loader))

    reward = val_psnr - 10.0 * val_loss

    train_loss /= max(1, len(train_loader))

    print()
    print(f"Epoch {epoch}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"PSNR       : {val_psnr:.2f}")
    print(f"Reward     : {reward:.2f}")
    print()

    # =====================
    # SAVE LAST
    # =====================

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

    # =====================
    # SAVE BEST
    # =====================

    if reward > best_reward:
        best_reward = reward

        torch.save(model.state_dict(), best_ckpt)

        print("NEW BEST MODEL SAVED")