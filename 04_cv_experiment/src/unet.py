"""U-Net קטן עם מקודד ResNet-18 מאומן מראש (ImageNet), לסגמנטציה של 5 מחלקות.

ארכיטקטורה (קלט 3xHxW, H ו-W מתחלקים ב-32):

  מקודד (torchvision resnet18, 11.2M פרמטרים)        מפענח (~3.2M פרמטרים)
  ---------------------------------------------      ----------------------------------------
  stem  conv7x7/2 + BN + ReLU      ->  64 x H/2   ─┐
  pool  maxpool/2                  ->  64 x H/4    │
  layer1                           ->  64 x H/4   ─┼┐
  layer2                           -> 128 x H/8   ─┼┼┐
  layer3                           -> 256 x H/16  ─┼┼┼┐
  layer4 (צוואר בקבוק)             -> 512 x H/32   ││││
                                                   ││││  up4: 512 + 256(skip) -> 256 x H/16
                                                   │││└─ up3: 256 + 128(skip) -> 128 x H/8
                                                   ││└── up2: 128 +  64(skip) ->  64 x H/4
                                                   │└─── up1:  64 +  64(stem) ->  64 x H/2
                                                   └──── head: upsample x2, conv3x3(32), conv1x1 -> 5 x H

כל בלוק מפענח: upsample בילינארי x2, שרשור skip, ושתי שכבות conv3x3 + BN + ReLU.
סה"כ כ-14.4M פרמטרים. באימון ב-FP16 עם batch 8 וחיתוכים של 384x384 נמדד על RTX 3050 6GB
שיא של 1,664MB לפי torch.cuda.max_memory_allocated, ו-1,901MiB לפי nvidia-smi (כולל הקשר CUDA).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(_conv_bn_relu(cin + cskip, cout), _conv_bn_relu(cout, cout))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNet18UNet(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 64, H/2
        self.pool = enc.maxpool                                    # 64, H/4
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)
        self.head = nn.Sequential(_conv_bn_relu(64, 32), nn.Conv2d(32, num_classes, 1))

    def encoder_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def decoder_parameters(self):
        for m in (self.up4, self.up3, self.up2, self.up1, self.head):
            yield from m.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)                 # H/2
        s1 = self.layer1(self.pool(s0))   # H/4
        s2 = self.layer2(s1)              # H/8
        s3 = self.layer3(s2)              # H/16
        s4 = self.layer4(s3)              # H/32
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(d)


def count_parameters(model: nn.Module) -> dict:
    enc = sum(p.numel() for p in model.encoder_parameters())
    dec = sum(p.numel() for p in model.decoder_parameters())
    return {"encoder": enc, "decoder": dec, "total": enc + dec}
