import os
import math
import random
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm

from torch.utils.data import DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import RGBDataset

from model import (
    build_generator,
    build_discriminator
)

from losses import (
    GeneratorLoss,
    DiscriminatorLoss
)

# ==========================================================
# CONFIG
# ==========================================================

DATASET_PATH = r"C:\Users\Vinay Singh\Desktop\OUTPUT_ARCHIVED\patches"

CHECKPOINT_DIR = "checkpoints"

SAMPLE_DIR = "samples"

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    SAMPLE_DIR,
    exist_ok=True
)

GENERATOR_BEST = os.path.join(
    CHECKPOINT_DIR,
    "generator_best.pth"
)

DISCRIMINATOR_BEST = os.path.join(
    CHECKPOINT_DIR,
    "discriminator_best.pth"
)

LAST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "last_checkpoint.pth"
)

# ==========================================================
# TRAINING CONFIG
# ==========================================================

EPOCHS = 100

BATCH_SIZE = 4

PATCH_SIZE = 256

LEARNING_RATE = 2e-4

BETA1 = 0.5

BETA2 = 0.999

L1_WEIGHT = 100.0

NUM_WORKERS = 0

PIN_MEMORY = True

SEED = 42

# ==========================================================
# DEVICE
# ==========================================================

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
# RANDOM SEED
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

# ==========================================================
# PSNR
# ==========================================================


def calculate_psnr(pred, target):

    mse = F.mse_loss(
        pred,
        target
    )

    if mse.item() == 0:

        return 100.0

    return 20 * math.log10(1.0) - 10 * math.log10(mse.item())

# ==========================================================
# SSIM
# ==========================================================


ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0
).to(DEVICE)

# ==========================================================
# DATASET
# ==========================================================

full_dataset = RGBDataset(

    DATASET_PATH,

    patch_size=PATCH_SIZE,

    training=True

)

random.shuffle(
    full_dataset.samples
)

split = int(
    0.9 * len(full_dataset.samples)
)

train_samples = full_dataset.samples[:split]

val_samples = full_dataset.samples[split:]

train_dataset = RGBDataset(

    DATASET_PATH,

    patch_size=PATCH_SIZE,

    training=True,

    samples=train_samples

)

val_dataset = RGBDataset(

    DATASET_PATH,

    patch_size=PATCH_SIZE,

    training=False,

    samples=val_samples

)

print()

print("=" * 60)

print("Training Samples :", len(train_dataset))

print("Validation Samples :", len(val_dataset))

print("=" * 60)

# ==========================================================
# DATALOADER
# ==========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY

)

# ==========================================================
# MODELS
# ==========================================================

generator = build_generator().to(
    DEVICE
)

discriminator = build_discriminator().to(
    DEVICE
)

print()

print("=" * 60)

print(
    "Generator Parameters :",
    f"{sum(p.numel() for p in generator.parameters() if p.requires_grad):,}"
)

print(
    "Discriminator Parameters :",
    f"{sum(p.numel() for p in discriminator.parameters() if p.requires_grad):,}"
)

print("=" * 60)





# ==========================================================
# LOSSES
# ==========================================================


generator_loss = GeneratorLoss(

    l1_weight=25.0,

    perceptual_weight=5.0,

    edge_weight=2.0

).to(DEVICE)



discriminator_loss = DiscriminatorLoss().to(
    DEVICE
)

# ==========================================================
# OPTIMIZERS
# ==========================================================

optimizer_G = AdamW(

    generator.parameters(),

    lr=LEARNING_RATE,

    betas=(BETA1, BETA2),

    weight_decay=1e-4

)

optimizer_D = AdamW(

    discriminator.parameters(),

    lr=LEARNING_RATE,

    betas=(BETA1, BETA2),

    weight_decay=1e-4

)

# ==========================================================
# LR SCHEDULERS
# ==========================================================

scheduler_G = CosineAnnealingLR(

    optimizer_G,

    T_max=EPOCHS,

    eta_min=1e-6

)

scheduler_D = CosineAnnealingLR(

    optimizer_D,

    T_max=EPOCHS,

    eta_min=1e-6

)

# ==========================================================
# MIXED PRECISION
# ==========================================================

USE_AMP = DEVICE.type == "cuda"

scaler_G = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)

scaler_D = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)

# ==========================================================
# RESUME TRAINING
# ==========================================================

start_epoch = 1

best_l1 = float("inf")

if os.path.exists(LAST_CHECKPOINT):

    print()

    print("=" * 60)

    print("Loading Checkpoint...")

    checkpoint = torch.load(

        LAST_CHECKPOINT,

        map_location=DEVICE

    )

    generator.load_state_dict(

        checkpoint["generator"]

    )

    discriminator.load_state_dict(

        checkpoint["discriminator"]

    )

    optimizer_G.load_state_dict(

        checkpoint["optimizer_G"]

    )

    optimizer_D.load_state_dict(

        checkpoint["optimizer_D"]

    )

    scheduler_G.load_state_dict(

        checkpoint["scheduler_G"]

    )

    scheduler_D.load_state_dict(

        checkpoint["scheduler_D"]

    )

    scaler_G.load_state_dict(

        checkpoint["scaler_G"]

    )

    scaler_D.load_state_dict(

        checkpoint["scaler_D"]

    )

    start_epoch = checkpoint["epoch"] + 1

    best_l1 = checkpoint["best_l1"]

    print("Checkpoint Loaded")

    print("Resume Epoch :", start_epoch)

    print("Best Validation L1 :", f"{best_l1:.6f}")

    print("=" * 60)

else:

    print()

    print("=" * 60)

    print("Starting New Training")

    print("=" * 60)

print()

print("Generator Optimizer : AdamW")

print("Discriminator Optimizer : AdamW")

print("Learning Rate :", LEARNING_RATE)

print("Mixed Precision :", USE_AMP)

print("=" * 60)

print()

# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(start_epoch, EPOCHS + 1):

    print()

    print("=" * 60)

    print(f"Epoch {epoch}/{EPOCHS}")

    print("=" * 60)

    generator.train()

    discriminator.train()

    running_g_loss = 0.0

    running_d_loss = 0.0

    running_l1 = 0.0

    train_bar = tqdm(

        train_loader,

        leave=True,

        dynamic_ncols=True

    )

    for thermal, real_rgb in train_bar:

        thermal = thermal.to(
            DEVICE,
            non_blocking=True
        )

        real_rgb = real_rgb.to(
            DEVICE,
            non_blocking=True
        )

        # ==================================================
        # TRAIN GENERATOR
        # ==================================================

        optimizer_G.zero_grad(set_to_none=True)

        with torch.autocast(

            device_type="cuda",

            dtype=torch.float16,

            enabled=USE_AMP

        ):

            fake_rgb = generator(
                thermal
            )

            pred_fake = discriminator(
                thermal,
                fake_rgb
            )

            real_label = torch.full_like(
                pred_fake,
                0.9
            )

            fake_label = torch.zeros_like(
                pred_fake
            )

            loss_G, loss_gan, loss_l1 = generator_loss(

                pred_fake,

                fake_rgb,

                real_rgb,

                real_label

            )

        scaler_G.scale(
            loss_G
        ).backward()

        scaler_G.step(
            optimizer_G
        )

        scaler_G.update()
        #fake_rgb = torch.clamp(fake_rgb, 0.0, 1.0)

        # ==================================================
        # TRAIN DISCRIMINATOR
        # ==================================================

        optimizer_D.zero_grad(set_to_none=True)

        with torch.autocast(

            device_type="cuda",

            dtype=torch.float16,

            enabled=USE_AMP

        ):

            pred_real = discriminator(

                thermal,

                real_rgb

            )

            pred_fake = discriminator(

                thermal,

                fake_rgb.detach()

            )

            loss_D, loss_real, loss_fake = discriminator_loss(

                pred_real,

                pred_fake,

                real_label,

                fake_label

            )

        scaler_D.scale(
            loss_D
        ).backward()

        scaler_D.step(
            optimizer_D
        )

        scaler_D.update()

        running_g_loss += loss_G.item()

        running_d_loss += loss_D.item()

        running_l1 += loss_l1.item()

        train_bar.set_postfix(

            G=f"{loss_G.item():.4f}",

            D=f"{loss_D.item():.4f}",

            L1=f"{loss_l1.item():.4f}"

        )

    running_g_loss /= len(train_loader)

    running_d_loss /= len(train_loader)

    running_l1 /= len(train_loader)

    scheduler_G.step()

    scheduler_D.step()

    # ==========================================================
# VALIDATION
# ==========================================================

    generator.eval()

    val_l1 = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    first_batch = True

    with torch.no_grad():

        for thermal, real_rgb in tqdm(

            val_loader,

            desc="Validation",

            leave=False,

            dynamic_ncols=True

        ):

            thermal = thermal.to(
                DEVICE,
                non_blocking=True
            )

            real_rgb = real_rgb.to(
                DEVICE,
                non_blocking=True
            )

            with torch.autocast(

                device_type="cuda",

                dtype=torch.float16,

                enabled=USE_AMP

            ):

                fake_rgb = generator(
                    thermal
                )

                loss = F.l1_loss(
                    fake_rgb,
                    real_rgb
                )

            val_l1 += loss.item()

            val_psnr += calculate_psnr(
                fake_rgb,
                real_rgb
            )

            val_ssim += ssim_metric(
                fake_rgb,
                real_rgb
            ).item()

            # ---------------------------------------------
            # SAVE FIRST VALIDATION IMAGE
            # ---------------------------------------------

            if first_batch:

                import torchvision.utils as vutils

                grid = torch.cat(

                    [

                        thermal.repeat(
                            1,
                            3,
                            1,
                            1
                        ),

                        fake_rgb,

                        real_rgb

                    ],

                    dim=0

                )

                vutils.save_image(

                    grid,

                    os.path.join(

                        SAMPLE_DIR,

                        f"epoch_{epoch:03d}.png"

                    ),

                    nrow=thermal.size(0),

                    normalize=True

                )

                first_batch = False

    val_l1 /= len(val_loader)

    val_psnr /= len(val_loader)

    val_ssim /= len(val_loader)

    print()

    print("=" * 60)

    print(f"Generator Loss     : {running_g_loss:.6f}")

    print(f"Discriminator Loss : {running_d_loss:.6f}")

    print(f"Train L1           : {running_l1:.6f}")

    print(f"Validation L1      : {val_l1:.6f}")

    print(f"Validation PSNR    : {val_psnr:.2f} dB")

    print(f"Validation SSIM    : {val_ssim:.4f}")

    print("=" * 60)

    print()

    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    if val_l1 < best_l1:

        best_l1 = val_l1

        torch.save(

            generator.state_dict(),

            GENERATOR_BEST

        )

        torch.save(

            discriminator.state_dict(),

            DISCRIMINATOR_BEST

        )

        print()

        print("★★★★★ NEW BEST MODEL SAVED ★★★★★")

        print()

    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================

    torch.save(

        {

            "epoch": epoch,

            "generator": generator.state_dict(),

            "discriminator": discriminator.state_dict(),

            "optimizer_G": optimizer_G.state_dict(),

            "optimizer_D": optimizer_D.state_dict(),

            "scheduler_G": scheduler_G.state_dict(),

            "scheduler_D": scheduler_D.state_dict(),

            "scaler_G": scaler_G.state_dict(),

            "scaler_D": scaler_D.state_dict(),

            "best_l1": best_l1

        },

        LAST_CHECKPOINT

    )

    print("Checkpoint Saved")

# ==========================================================
# TRAINING FINISHED
# ==========================================================

print()

print("=" * 60)

print("TRAINING FINISHED")

print()

print(f"Best Validation L1 : {best_l1:.6f}")

print()

print("Generator :", GENERATOR_BEST)

print("Discriminator :", DISCRIMINATOR_BEST)

print("Checkpoint :", LAST_CHECKPOINT)

print("=" * 60)

torch.cuda.empty_cache()
