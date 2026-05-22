"""
anchors.py
----------
Anchor generation, box encoding/decoding, and IoU-based anchor matching.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
import config


# ── Anchor generation ─────────────────────────────────────────────────────────

def generate_student_anchors(device: torch.device) -> torch.Tensor:
    """
    Generate all anchors for the student detector.
    Returns [A_total, 4] in (x1, y1, x2, y2) format.
    """
    all_anchors = []
    for fsize, stride, sizes in [
        (18, config.HEAD1_STRIDE, config.HEAD1_ANCHOR_SIZES),
        ( 9, config.HEAD2_STRIDE, config.HEAD2_ANCHOR_SIZES),
    ]:
        for sz in sizes:
            sx = (torch.arange(fsize, device=device) + 0.5) * stride
            sy = (torch.arange(fsize, device=device) + 0.5) * stride
            cy, cx = torch.meshgrid(sy, sx, indexing="ij")
            cx, cy = cx.reshape(-1), cy.reshape(-1)
            half = sz / 2.0
            all_anchors.append(
                torch.stack([cx - half, cy - half, cx + half, cy + half], dim=1))
    return torch.cat(all_anchors, dim=0)


def generate_teacher_anchors(device: torch.device, img_size: int = 640) -> torch.Tensor:
    """Generate anchors for all four RetinaFace FPN levels."""
    all_anchors = []
    for level, cfg in config.TEACHER_ANCHOR_CFG.items():
        stride = cfg["stride"]
        fsize  = img_size // stride
        for sz in cfg["sizes"]:
            sx = (torch.arange(fsize, device=device) + 0.5) * stride
            sy = (torch.arange(fsize, device=device) + 0.5) * stride
            cy, cx = torch.meshgrid(sy, sx, indexing="ij")
            cx, cy = cx.reshape(-1), cy.reshape(-1)
            half = sz / 2.0
            all_anchors.append(
                torch.stack([cx - half, cy - half, cx + half, cy + half], dim=1))
    return torch.cat(all_anchors, dim=0)


# ── Box coder ─────────────────────────────────────────────────────────────────

WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes as deltas relative to anchors."""
    wa  = anchors[:, 2] - anchors[:, 0]
    ha  = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + 0.5 * wa
    cya = anchors[:, 1] + 0.5 * ha

    wb  = gt_boxes[:, 2] - gt_boxes[:, 0]
    hb  = gt_boxes[:, 3] - gt_boxes[:, 1]
    cxb = gt_boxes[:, 0] + 0.5 * wb
    cyb = gt_boxes[:, 1] + 0.5 * hb

    return torch.stack([
        WEIGHTS[0] * (cxb - cxa) / wa,
        WEIGHTS[1] * (cyb - cya) / ha,
        WEIGHTS[2] * torch.log(wb  / wa),
        WEIGHTS[3] * torch.log(hb  / ha),
    ], dim=1)


def decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Decode predicted deltas back to (x1, y1, x2, y2) boxes."""
    wa  = anchors[:, 2] - anchors[:, 0]
    ha  = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + 0.5 * wa
    cya = anchors[:, 1] + 0.5 * ha

    cx = deltas[:, 0] / WEIGHTS[0] * wa  + cxa
    cy = deltas[:, 1] / WEIGHTS[1] * ha  + cya
    w  = torch.exp(deltas[:, 2] / WEIGHTS[2]) * wa
    h  = torch.exp(deltas[:, 3] / WEIGHTS[3]) * ha

    return torch.stack([cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h], dim=1)


# ── Anchor matcher ────────────────────────────────────────────────────────────

def match_anchors(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    pos_thr: float = config.POS_IOU_THRESH,
    neg_thr: float = config.NEG_IOU_THRESH,
):
    """
    Assign each anchor a label:
        +1  positive (IoU ≥ pos_thr with some GT box)
         0  negative / background (IoU < neg_thr)
        -1  ignore (between thresholds)

    Every GT box is guaranteed to match at least one anchor.

    Returns:
        matched_idxs : [A] index into gt_boxes for each positive anchor
        labels       : [A] {-1, 0, 1}
    """
    A = len(anchors)
    if len(gt_boxes) == 0:
        return (
            torch.full((A,), -1, dtype=torch.long, device=anchors.device),
            torch.zeros(A, dtype=torch.long, device=anchors.device),
        )

    iou_matrix = box_iou(anchors, gt_boxes)          # [A, G]
    max_iou, matched_idxs = iou_matrix.max(dim=1)    # [A]

    labels = torch.full((A,), -1, dtype=torch.long, device=anchors.device)
    labels[max_iou >= pos_thr] = 1
    labels[max_iou <  neg_thr] = 0

    # Force every GT to have ≥1 anchor
    best_anchor_per_gt = iou_matrix.argmax(dim=0)    # [G]
    labels[best_anchor_per_gt] = 1
    matched_idxs[best_anchor_per_gt] = torch.arange(
        len(gt_boxes), device=gt_boxes.device)

    return matched_idxs, labels
