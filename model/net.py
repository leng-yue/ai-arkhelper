import torch
import numpy as np
from torch import nn

from base import ACTIONS
from model.mbv3 import mbv3_small


# Conv BatchNorm Activation
class CBAModule(nn.Module):
    def __init__(self, in_channels, out_channels=24, kernel_size=3, stride=1, padding=0, bias=False, activation=None):
        super(CBAModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Up Sample Module
class UpSampleModule(nn.Module):
    """
    上采样
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, bias=False, mode="UCBA"):
        super(UpSampleModule, self).__init__()
        self.mode = mode

        if self.mode == "UCBA":
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = CBAModule(in_channels, out_channels, 3, padding=1, bias=bias)
        elif self.mode == "DeconvBN":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        elif self.mode == "DeCBA":
            self.dconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, bias=bias)
            self.conv = CBAModule(out_channels, out_channels, 3, padding=1, bias=bias)
        else:
            raise RuntimeError(f"Unsupport mode: {mode}")

    def forward(self, x):
        if self.mode == "UCBA":
            return self.conv(self.up(x))
        elif self.mode == "DeconvBN":
            return self.relu(self.bn(self.dconv(x)))
        elif self.mode == "DeCBA":
            return self.conv(self.dconv(x))


# SSH Context Module
class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()

        block_wide = in_channels // 4
        self.inconv = CBAModule(in_channels, block_wide, 3, 1, padding=1)
        self.upconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv = CBAModule(block_wide, block_wide, 3, 1, padding=1)
        self.downconv2 = CBAModule(block_wide, block_wide, 3, 1, padding=1)

    def forward(self, x):
        x = self.inconv(x)
        up = self.upconv(x)
        down = self.downconv(x)
        down = self.downconv2(down)
        return torch.cat([up, down], dim=1)


# SSH Detect Module
class DetectModule(nn.Module):
    def __init__(self, in_channels):
        super(DetectModule, self).__init__()

        self.upconv = CBAModule(in_channels, in_channels // 2, 3, 1, padding=1)
        self.context = ContextModule(in_channels)

    def forward(self, x):
        up = self.upconv(x)
        down = self.context(x)
        return torch.cat([up, down], dim=1)


# Job Head Module
class HeadModule(nn.Module):
    def __init__(self, in_channels, out_channels, has_ext=False):
        super(HeadModule, self).__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.has_ext = has_ext

        if has_ext:
            self.ext = CBAModule(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def init_normal(self, std, bias):
        nn.init.normal_(self.head.weight, std=std)
        nn.init.constant_(self.head.bias, bias)

    def forward(self, x):

        if self.has_ext:
            x = self.ext(x)
        return self.head(x)


# CenterNet Model
class ArkNet(nn.Module):
    def __init__(self, wide=64, has_ext=True, up_mode="UCBA"):
        super(ArkNet, self).__init__()

        c0, c1, c2 = [16, 24, 48]
        conv3_dim = 96
        num_layers = len(ACTIONS) + 3
        self.backbone = mbv3_small(keep=[1, 3, 8, 10], run_to=11, num_layers=num_layers)
        self.conv3 = CBAModule(conv3_dim, wide, kernel_size=1, stride=1, padding=0, bias=False)  # s32
        self.connect0 = CBAModule(c0, wide, kernel_size=1, stride=1)  # s4
        self.connect1 = CBAModule(c1, wide, kernel_size=1)  # s8
        self.connect2 = CBAModule(c2, wide, kernel_size=1)  # s16

        self.up0 = UpSampleModule(wide, wide, kernel_size=2, stride=2, mode=up_mode)  # s16
        self.up1 = UpSampleModule(wide, wide, kernel_size=2, stride=2, mode=up_mode)  # s8
        self.up2 = UpSampleModule(wide, wide, kernel_size=2, stride=2, mode=up_mode)  # s4
        self.detect = DetectModule(wide)

        self.finished = nn.Sequential(
            CBAModule(conv3_dim, 256, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 2)
        )
        self.center = HeadModule(wide, 1, has_ext=has_ext)
        self.box = HeadModule(wide, 4, has_ext=has_ext)

        self.init_weights()

    def init_weights(self):
        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19
        # Load backbone weights from ImageNet
        self.center.init_normal(0.001, d)
        self.box.init_normal(0.001, 0)

    def forward(self, x):
        s4, s8, s16, s32 = self.backbone(x)
        finished = self.finished(s32)

        s32 = self.conv3(s32)
        s16 = self.up0(s32) + self.connect2(s16)
        s8 = self.up1(s16) + self.connect1(s8)
        s4 = self.up2(s8) + self.connect0(s4)
        x = self.detect(s4)

        center = self.center(x)
        box = self.box(x)

        center = center.sigmoid()
        box = torch.exp(box)
        return finished, center, box
