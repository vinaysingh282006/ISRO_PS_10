import torch
import torch.nn as nn


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y


class RCAB(nn.Module):
    def __init__(self, channels):

        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            CALayer(channels)
        )

    def forward(self, x):

        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, channels, num_blocks):

        super().__init__()

        blocks = []

        for _ in range(num_blocks):
            blocks.append(RCAB(channels))

        blocks.append(
            nn.Conv2d(channels, channels, 3, padding=1)
        )

        self.body = nn.Sequential(*blocks)

    def forward(self, x):

        return x + self.body(x)


class RCANLite(nn.Module):

    def __init__(
        self,
        channels=64,
        num_groups=8,
        num_blocks=12,
        scale=2
    ):
        super().__init__()

        self.head = nn.Conv2d(
            1,
            channels,
            3,
            padding=1
        )

        groups = []

        for _ in range(num_groups):
            groups.append(
                ResidualGroup(
                    channels,
                    num_blocks
                )
            )

        groups.append(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

        self.body = nn.Sequential(*groups)

        self.upsample = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bicubic",
                align_corners=False
            ),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                1,
                3,
                padding=1
            )
        )

    def forward(self, x):

        shallow = self.head(x)

        deep = self.body(shallow)

        feat = shallow + deep

        out = self.upsample(feat)

        return out


if __name__ == "__main__":

    model = RCANLite()

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {total:,}"
    )
