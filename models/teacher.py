"""
models/teacher.py
-----------------
RetinaFace with ResNet-50 backbone.

Deng et al., "RetinaFace: Single-Shot Multi-Level Face Localisation
in the Wild", CVPR 2020.

This model is the TEACHER in the KD pipeline.
It is NOT deployable on Coral or MAX78000 due to:
  - ResNet-50 residual connections across spatial sizes
  - FPN cross-scale feature merging
  - Model size: ~118 MB FP32
Its only role is to produce rich feature maps and soft predictions
that the student learns from.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SSH(nn.Module):
    """Single Stage Headless context module."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert out_ch % 4 == 0
        half, qtr = out_ch // 2, out_ch // 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, half, 3, 1, 1, bias=False), nn.BatchNorm2d(half))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, qtr, 3, 1, 1, bias=False), nn.BatchNorm2d(qtr), nn.ReLU(True),
            nn.Conv2d(qtr,  qtr, 3, 1, 1, bias=False), nn.BatchNorm2d(qtr))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_ch, qtr, 3, 1, 1, bias=False), nn.BatchNorm2d(qtr), nn.ReLU(True),
            nn.Conv2d(qtr,  qtr, 3, 1, 1, bias=False), nn.BatchNorm2d(qtr), nn.ReLU(True),
            nn.Conv2d(qtr,  qtr, 3, 1, 1, bias=False), nn.BatchNorm2d(qtr))
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], 1))


class FPN(nn.Module):
    """Top-down FPN with lateral connections."""
    def __init__(self, in_channels_list, out_channels: int):
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.smooth  = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, 1, 1) for _ in in_channels_list])

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode="nearest")
        return [s(l) for s, l in zip(self.smooth, laterals)]


class DetHead(nn.Module):
    def __init__(self, in_ch: int, n_anchors: int):
        super().__init__()
        self.cls = nn.Conv2d(in_ch, n_anchors * 2, 1)
        self.reg = nn.Conv2d(in_ch, n_anchors * 4, 1)
        nn.init.constant_(self.cls.bias[:n_anchors], -math.log(99))
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.zeros_(self.reg.bias)
        nn.init.normal_(self.reg.weight, std=0.01)

    def forward(self, x):
        return self.cls(x), self.reg(x)


class RetinaFace(nn.Module):
    """
    RetinaFace ResNet-50.
    FPN channels : 256
    Anchors      : 2 per FPN level (P2–P5)
    Input        : 640×640
    """

    FPN_CH    = 256
    N_ANCHORS = 2

    def __init__(self, pretrained: bool = True):
        super().__init__()

        r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.layer0 = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.layer1 = r50.layer1   # 256 ch, stride 4
        self.layer2 = r50.layer2   # 512 ch, stride 8   → C2
        self.layer3 = r50.layer3   # 1024 ch, stride 16 → C3
        self.layer4 = r50.layer4   # 2048 ch, stride 32 → C4

        # Extra stride-2 conv for C5
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True))

        self.fpn = FPN(
            in_channels_list=[512, 1024, 2048, 256],
            out_channels=self.FPN_CH)

        self.ssh  = nn.ModuleDict(
            {k: SSH(self.FPN_CH, self.FPN_CH) for k in ["P2", "P3", "P4", "P5"]})
        self.head = nn.ModuleDict(
            {k: DetHead(self.FPN_CH, self.N_ANCHORS) for k in ["P2", "P3", "P4", "P5"]})

        # Adapter layers: project FPN features to student channel depth (64)
        # Used only during distillation — not part of the deployed model
        self.adapt = nn.ModuleDict({
            "P3": nn.Conv2d(self.FPN_CH, 64, 1),
            "P4": nn.Conv2d(self.FPN_CH, 64, 1),
        })

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x  = self.layer0(x)
        x  = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])

        feats, preds = {}, []
        for feat, key in [(p2, "P2"), (p3, "P3"), (p4, "P4"), (p5, "P5")]:
            feat = self.ssh[key](feat)
            feats[key] = feat
            preds.append(self.head[key](feat))

        if return_features:
            adapted = {k: self.adapt[k](feats[k]) for k in ["P3", "P4"]}
            return preds, adapted
        return preds


def load_teacher(weights_path: str, device: torch.device) -> RetinaFace:
    """Load RetinaFace with pretrained WIDER FACE weights."""
    teacher = RetinaFace(pretrained=True).to(device)

    import os
    if os.path.exists(weights_path):
        ckpt  = torch.load(weights_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = teacher.load_state_dict(state, strict=False)
        if missing:
            print(f"  [Teacher] Missing keys   : {len(missing)}")
        if unexpected:
            print(f"  [Teacher] Unexpected keys: {len(unexpected)}")
        print(f"  [Teacher] Loaded pretrained RetinaFace weights from {weights_path}")
    else:
        print(f"  [Teacher] ⚠️  No weights at {weights_path} — using ImageNet init only")
        print("             Run download_dataset.py to get pretrained weights.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
