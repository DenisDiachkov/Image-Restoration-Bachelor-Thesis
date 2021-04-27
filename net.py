from pytorch_lightning import LightningModule
import torch
import torch.nn as nn


class SampleModel(LightningModule):
    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 3,
        **kwargs
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_features, 8, 3, padding=1)

        self.conv2 = torch.nn.Conv2d(
            8, out_features, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ResidualBlock(nn.Module):
    def __init__(
        self, ch:int
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return out + x


class DownscalingBlock(nn.Module):
    def __init__(
        self, in_ch:int, out_ch:int, downscale=True
    ):
        super().__init__()
        self.downconv = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2 if downscale else 1, padding=2)
        self.resnet = nn.Sequential(
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
        )
    
    def forward(self, x):
        return self.resnet(self.downconv(x))

class UpscalingBlock(nn.Module):
    def __init__(
        self, in_ch:int, out_ch:int, upscale=True
    ):
        super().__init__()
        self.resnet = nn.Sequential(
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
            ResidualBlock(in_ch),
        )
        self.upconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4 if upscale else 5, stride=2 if upscale else 1, padding=1 if upscale else 2)   
    
    def forward(self, x):
        return self.upconv(self.resnet(x))


class ImageRestorationModel(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
    ):
        super().__init__()
        self.down1 = DownscalingBlock(in_ch, 32, False)
        self.down2 = DownscalingBlock(32, 64)
        self.down3 = DownscalingBlock(64, 128)

        self.up1 = UpscalingBlock(128, 64)
        self.up2 = UpscalingBlock(64, 32)
        self.up3 = UpscalingBlock(32, 16, False)
        self.up4 = UpscalingBlock(16, 8)
        self.up5 = UpscalingBlock(8, out_ch)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        x4 = self.up1(x3)
        x5 = self.up2(x4+x2)
        x6 = self.up3(x5+x1)
        x7 = self.up4(x6)
        x8 = self.up5(x7)
        return x8
