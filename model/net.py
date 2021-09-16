import torch
import numpy as np
from torch import nn

from definition import TASKS, ACTIONS
from efficientnet_pytorch import EfficientNet


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

        c0, c1, c2 = [16, 24, 40]
        embedding_cls = 32
        conv3_dim = 112 + embedding_cls
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv3 = CBAModule(conv3_dim, wide, kernel_size=1, stride=1, padding=0, bias=False)  # s32
        self.connect0 = CBAModule(c0, wide, kernel_size=1, stride=2)  # s4
        self.connect1 = CBAModule(c1, wide, kernel_size=1)  # s8
        self.connect2 = CBAModule(c2, wide, kernel_size=1)  # s16

        self.up0 = UpSampleModule(wide, wide, kernel_size=2, stride=2, mode=up_mode)  # s16
        self.up1 = UpSampleModule(wide, wide, kernel_size=2, stride=2, mode=up_mode)  # s8
        self.up2 = UpSampleModule(wide, wide, kernel_size=2, stride=2, mode=up_mode)  # s4
        self.detect = DetectModule(wide)

        self.task_embedding = nn.Sequential(
            nn.Linear(len(TASKS), embedding_cls),
            nn.ReLU()
        )

        self.action = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280 + embedding_cls, len(TASKS) * len(ACTIONS))
        )

        self.center = HeadModule(wide, len(TASKS), has_ext=has_ext)
        self.box = HeadModule(wide, 4, has_ext=has_ext)

        self.init_weights()

    def init_weights(self):
        # Set the initial probability to avoid overflow at the beginning
        prob = 0.01
        d = -np.log((1 - prob) / prob)  # -2.19
        # Load backbone weights from ImageNet
        self.center.init_normal(0.001, d)
        self.box.init_normal(0.001, 0)

    def forward(self, x, task_encoding):
        s16, s24, s40, s112, _, feature = self.backbone.extract_endpoints(x).values()

        task = self.task_embedding(task_encoding).unsqueeze(2).unsqueeze(3)
        s112 = torch.cat([s112, task.repeat(1, 1, 16, 16)], 1)
        feature = torch.cat([feature, task.repeat(1, 1, 8, 8)], 1)
        action = self.action(feature)

        s112 = self.conv3(s112)
        s40 = self.up0(s112) + self.connect2(s40)
        s24 = self.up1(s40) + self.connect1(s24)
        s16 = s24 + self.connect0(s16)
        x = self.detect(s16)

        center = self.center(x)
        box = self.box(x)

        center = center.sigmoid()
        box = torch.exp(box)
        return action, center, box
