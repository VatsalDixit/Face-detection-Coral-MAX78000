"""
models/student.py
-----------------
MAX78000 and Coral Edge TPU compatible face detector.
Trained via Knowledge Distillation from RetinaFace ResNet-50.

Hardware compatibility:
  MAX78000 : ✅ standard Conv2d + MaxPool, ≤64 channels, 74×74 input
  Coral TPU: ✅ TFLite INT8, Conv2d, MaxPool fully supported

Architecture (74×74 RGB input):
  conv1  3 → 16    74×74   [streamed via FIFO on MAX78000]
  pool1           →37×37
  conv2  16→ 32   37×37
  pool2           →18×18
  conv3  32→ 64   18×18
  conv4  64→ 64   18×18   ← detection head 1 (larger faces)
  pool3           → 9×9
  conv5  64→ 64    9×9
  conv6  64→ 64    9×9    ← detection head 2 (smaller faces)

Anchor sizes:
  Head1 (18×18): [6, 10, 16] px  — stride ≈4.1
  Head2  (9×9):  [22, 32, 46] px — stride ≈8.2
"""

import math
import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    """Standard Conv2d + BN + ReLU.  Only uses ops supported by both MAX78000 and Coral."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StudentDetector(nn.Module):
    """
    Hardware-legal face detector.
    Distilled from RetinaFace ResNet-50 on WIDER FACE.
    """

    IMG_SIZE = 74
    N_ANCHORS = 3

    # Anchor config — sizes in 74×74 image space
    HEAD1_ANCHOR_SIZES = [6,  10, 16]
    HEAD2_ANCHOR_SIZES = [22, 32, 46]
    HEAD1_STRIDE       = 74 / 18   # ≈ 4.11
    HEAD2_STRIDE       = 74 / 9    # ≈ 8.22

    def __init__(self):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.conv1 = ConvBnRelu(3,  16)
        self.pool1 = nn.MaxPool2d(2, 2)    # 74 → 37
        self.conv2 = ConvBnRelu(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)    # 37 → 18
        self.conv3 = ConvBnRelu(32, 64)
        self.conv4 = ConvBnRelu(64, 64)    # ← Head 1 source
        self.pool3 = nn.MaxPool2d(2, 2)    # 18 → 9
        self.conv5 = ConvBnRelu(64, 64)
        self.conv6 = ConvBnRelu(64, 64)    # ← Head 2 source

        # ── Detection heads ───────────────────────────────────────────────────
        # 1×1 conv — lightweight, supported on both platforms
        self.cls1 = nn.Conv2d(64, self.N_ANCHORS,     1)  # [B, 3, 18, 18]
        self.reg1 = nn.Conv2d(64, self.N_ANCHORS * 4, 1)  # [B,12, 18, 18]
        self.cls2 = nn.Conv2d(64, self.N_ANCHORS,     1)  # [B, 3,  9,  9]
        self.reg2 = nn.Conv2d(64, self.N_ANCHORS * 4, 1)  # [B,12,  9,  9]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Face-class bias: start with low prior to avoid exploding focal loss
        for h in [self.cls1, self.cls2]:
            nn.init.constant_(h.bias, -math.log(99))   # prior ≈ 0.01

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x     = self.pool1(self.conv1(x))
        x     = self.pool2(self.conv2(x))
        x     = self.conv3(x)
        feat1 = self.conv4(x)              # 64 × 18 × 18
        x     = self.pool3(feat1)
        x     = self.conv5(x)
        feat2 = self.conv6(x)              # 64 × 9 × 9

        preds = [
            (self.cls1(feat1), self.reg1(feat1)),
            (self.cls2(feat2), self.reg2(feat2)),
        ]

        if return_features:
            return preds, {"feat1": feat1, "feat2": feat2}
        return preds

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def size_kb_int8(self) -> float:
        return self.param_count / 1024
