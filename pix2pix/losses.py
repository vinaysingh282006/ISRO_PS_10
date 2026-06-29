import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    vgg19,
    VGG19_Weights
)

# ==========================================================
# VGG19 PERCEPTUAL LOSS
# ==========================================================

class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = vgg19(

            weights=VGG19_Weights.DEFAULT

        ).features[:18]

        for p in vgg.parameters():

            p.requires_grad = False

        self.vgg = vgg.eval()

        self.l1 = nn.L1Loss()

    def forward(

        self,

        fake,

        real

    ):

        fake = (fake - 0.5) * 2.0
        real = (real - 0.5) * 2.0

        fake_feature = self.vgg(fake)

        real_feature = self.vgg(real)

        return self.l1(

            fake_feature,

            real_feature

        )

# ==========================================================
# EDGE LOSS
# ==========================================================

class EdgeLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def sobel(self, x):

        gx = torch.tensor(

            [[-1,0,1],
             [-2,0,2],
             [-1,0,1]],

            dtype=x.dtype,

            device=x.device

        ).view(1,1,3,3)

        gy = torch.tensor(

            [[-1,-2,-1],
             [0,0,0],
             [1,2,1]],

            dtype=x.dtype,

            device=x.device

        ).view(1,1,3,3)

        edges = []

        for c in range(x.shape[1]):

            img = x[:,c:c+1]

            ex = F.conv2d(

                img,

                gx,

                padding=1

            )

            ey = F.conv2d(

                img,

                gy,

                padding=1

            )

            edges.append(

                torch.sqrt(

                    ex**2 + ey**2 + 1e-6

                )

            )

        return torch.cat(edges,1)

    def forward(

        self,

        fake,

        real

    ):

        return self.l1(

            self.sobel(fake),

            self.sobel(real)

        )

# ==========================================================
# GAN LOSS
# ==========================================================

class GANLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(

        self,

        prediction,

        target

    ):

        return self.loss(

            prediction,

            target

        )
    

# ==========================================================
# PIX2PIX GENERATOR LOSS
# ==========================================================

class GeneratorLoss(nn.Module):

    def __init__(

        self,

        l1_weight=10.0,

        perceptual_weight=10.0,

        edge_weight=5.0

    ):

        super().__init__()

        self.gan = GANLoss()

        self.l1 = nn.L1Loss()

        self.perceptual = PerceptualLoss()

        self.edge = EdgeLoss()

        self.l1_weight = l1_weight

        self.perceptual_weight = perceptual_weight

        self.edge_weight = edge_weight

    def forward(

        self,

        pred_fake,

        fake_rgb,

        real_rgb,

        real_label

    ):

        loss_gan = self.gan(

            pred_fake,

            real_label

        )

        loss_l1 = self.l1(

            fake_rgb,

            real_rgb

        )

        loss_perceptual = self.perceptual(

            fake_rgb,

            real_rgb

        )

        loss_edge = self.edge(

            fake_rgb,

            real_rgb

        )

        total = (

            loss_gan

            +

            self.l1_weight * loss_l1

            +

            self.perceptual_weight * loss_perceptual

            +

            self.edge_weight * loss_edge

        )

        return (

            total,

            loss_gan,

            loss_l1

        )


# ==========================================================
# PIX2PIX DISCRIMINATOR LOSS
# ==========================================================

class DiscriminatorLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.gan = GANLoss()

    def forward(

        self,

        pred_real,

        pred_fake,

        real_label,

        fake_label

    ):

        loss_real = self.gan(

            pred_real,

            real_label

        )

        loss_fake = self.gan(

            pred_fake,

            fake_label

        )

        total = (

            loss_real

            +

            loss_fake

        ) * 0.5

        return (

            total,

            loss_real,

            loss_fake

        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred_real = torch.randn(
        2, 1, 30, 30,
        device=device
    )

    pred_fake = torch.randn(
        2, 1, 30, 30,
        device=device
    )

    fake_rgb = torch.rand(
        2, 3, 256, 256,
        device=device
    )

    real_rgb = torch.rand(
        2, 3, 256, 256,
        device=device
    )

    real_label = torch.ones_like(pred_real)

    fake_label = torch.zeros_like(pred_fake)

    G = GeneratorLoss().to(device)

    D = DiscriminatorLoss().to(device)

    g_total, g_gan, g_l1 = G(

        pred_fake,

        fake_rgb,

        real_rgb,

        real_label

    )

    d_total, d_real, d_fake = D(

        pred_real,

        pred_fake,

        real_label,

        fake_label

    )

    print("=" * 60)

    print("Generator Total :", g_total.item())
    print("GAN            :", g_gan.item())
    print("L1             :", g_l1.item())

    print()

    print("Discriminator  :", d_total.item())

    print("=" * 60)