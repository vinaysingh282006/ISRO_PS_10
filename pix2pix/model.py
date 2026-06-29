import torch
import torch.nn as nn


# ==========================================================
# WEIGHT INITIALIZATION
# ==========================================================

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        nn.init.normal_(
            m.weight.data,
            0.0,
            0.02
        )

        if m.bias is not None:

            nn.init.zeros_(
                m.bias.data
            )

    elif classname.find("BatchNorm") != -1:

        nn.init.normal_(
            m.weight.data,
            1.0,
            0.02
        )

        nn.init.zeros_(
            m.bias.data
        )


# ==========================================================
# DOWN BLOCK
# ==========================================================

class DownBlock(nn.Module):

    def __init__(

        self,

        in_channels,

        out_channels,

        normalize=True

    ):

        super().__init__()

        layers = [

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=4,

                stride=2,

                padding=1,

                bias=False

            )

        ]

        if normalize:

            layers.append(

                nn.InstanceNorm2d(

                    out_channels

                )

            )

        layers.append(

            nn.LeakyReLU(

                0.2,

                inplace=True

            )

        )

        self.block = nn.Sequential(

            *layers

        )

    def forward(self, x):

        return self.block(x)


# ==========================================================
# UP BLOCK
# ==========================================================

class UpBlock(nn.Module):

    def __init__(

        self,

        in_channels,

        out_channels,

        dropout=False

    ):

        super().__init__()

        layers = [

            nn.ConvTranspose2d(

                in_channels,

                out_channels,

                kernel_size=4,

                stride=2,

                padding=1,

                bias=False

            ),

            nn.InstanceNorm2d(

                out_channels

            ),

            nn.ReLU(

                inplace=True

            )

        ]

        if dropout:

            layers.append(

                nn.Dropout(

                    0.5

                )

            )

        self.block = nn.Sequential(

            *layers

        )

    def forward(

        self,

        x,

        skip

    ):

        x = self.block(x)

        x = torch.cat(

            [

                x,

                skip

            ],

            dim=1

        )

        return x


# ==========================================================
# GENERATOR (PIX2PIX U-NET)
# ==========================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # ENCODER
        # --------------------------------------------------

        self.down1 = DownBlock(
            1,
            64,
            normalize=False
        )

        self.down2 = DownBlock(
            64,
            128
        )

        self.down3 = DownBlock(
            128,
            256
        )

        self.down4 = DownBlock(
            256,
            512
        )

        self.down5 = DownBlock(
            512,
            512
        )

        self.down6 = DownBlock(
            512,
            512
        )

        self.down7 = DownBlock(
            512,
            512
        )

        self.down8 = DownBlock(
            512,
            512,
            normalize=False
        )

        # --------------------------------------------------
        # DECODER
        # --------------------------------------------------

        self.up1 = UpBlock(
            512,
            512,
            dropout=True
        )

        self.up2 = UpBlock(
            1024,
            512,
            dropout=True
        )

        self.up3 = UpBlock(
            1024,
            512,
            dropout=True
        )

        self.up4 = UpBlock(
            1024,
            512
        )

        self.up5 = UpBlock(
            1024,
            256
        )

        self.up6 = UpBlock(
            512,
            128
        )

        self.up7 = UpBlock(
            256,
            64
        )

        # --------------------------------------------------
        # FINAL UPSAMPLING
        # --------------------------------------------------

        self.final = nn.Sequential(

            nn.ConvTranspose2d(

                128,

                3,

                kernel_size=4,

                stride=2,

                padding=1

                ),

                nn.Sigmoid()

)

    # ------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------

    def forward(self, x):

        d1 = self.down1(x)

        d2 = self.down2(d1)

        d3 = self.down3(d2)

        d4 = self.down4(d3)

        d5 = self.down5(d4)

        d6 = self.down6(d5)

        d7 = self.down7(d6)

        bottleneck = self.down8(d7)

        u1 = self.up1(
            bottleneck,
            d7
        )

        u2 = self.up2(
            u1,
            d6
        )

        u3 = self.up3(
            u2,
            d5
        )

        u4 = self.up4(
            u3,
            d4
        )

        u5 = self.up5(
            u4,
            d3
        )

        u6 = self.up6(
            u5,
            d2
        )

        u7 = self.up7(
            u6,
            d1
        )

        output = self.final(
            u7
        )

        return output



# ==========================================================
# PATCHGAN DISCRIMINATOR
# ==========================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        def block(
            in_channels,
            out_channels,
            normalize=True
        ):

            layers = [

                nn.Conv2d(

                    in_channels,

                    out_channels,

                    kernel_size=4,

                    stride=2,

                    padding=1,

                    bias=False

                )

            ]

            if normalize:

                layers.append(

                    nn.InstanceNorm2d(

                        out_channels

                    )

                )

            layers.append(

                nn.LeakyReLU(

                    0.2,

                    inplace=True

                )

            )

            return layers

        self.model = nn.Sequential(

            *block(
                4,
                64,
                normalize=False
            ),

            *block(
                64,
                128
            ),

            *block(
                128,
                256
            ),

            nn.Conv2d(

                256,

                512,

                kernel_size=4,

                stride=1,

                padding=1,

                bias=False

            ),

            nn.InstanceNorm2d(
                512
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(

                512,

                1,

                kernel_size=4,

                stride=1,

                padding=1

            )

        )

    def forward(

        self,

        thermal,

        rgb

    ):

        x = torch.cat(

            [

                thermal,

                rgb

            ],

            dim=1

        )

        return self.model(x)


# ==========================================================
# BUILD FUNCTIONS
# ==========================================================

def build_generator():

    model = Generator()

    model.apply(
        weights_init
    )

    return model


def build_discriminator():

    model = Discriminator()

    model.apply(
        weights_init
    )

    return model


# ==========================================================
# PARAMETER COUNT
# ==========================================================

def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )

    G = build_generator().to(device)

    D = build_discriminator().to(device)

    thermal = torch.randn(

        2,

        1,

        256,

        256

    ).to(device)

    rgb = torch.randn(

        2,

        3,

        256,

        256

    ).to(device)

    with torch.no_grad():

        fake_rgb = G(
            thermal
        )

        decision = D(
            thermal,
            fake_rgb
        )

    print("=" * 60)

    print("Pix2Pix Model Built Successfully")

    print("=" * 60)

    print()

    print(
        "Input Thermal :",
        thermal.shape
    )

    print(
        "Generated RGB :",
        fake_rgb.shape
    )

    print(
        "PatchGAN Out  :",
        decision.shape
    )

    print()

    print(

        "Generator Parameters :",

        f"{count_parameters(G):,}"

    )

    print(

        "Discriminator Parameters :",

        f"{count_parameters(D):,}"

    )

    print()

    print(

        "Total Parameters :",

        f"{count_parameters(G)+count_parameters(D):,}"

    )

    print("=" * 60)