"""
losses.py
---------
Focal Loss, GIoU Loss, and the full Knowledge Distillation loss
combining hard detection, feature matching, and soft-label distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from anchors import match_anchors, decode_boxes
import config


# ── Focal Loss ────────────────────────────────────────────────────────────────

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss. logits and targets are both [N]."""
    p    = torch.sigmoid(logits)
    ce   = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p_t  = p * targets + (1 - p) * (1 - targets)
    a_t  = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).sum()


# ── GIoU Loss ─────────────────────────────────────────────────────────────────

def giou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss for positive anchors."""
    if len(pred_boxes) == 0:
        return pred_boxes.sum() * 0
    return (1 - torch.diag(generalized_box_iou(pred_boxes, gt_boxes))).sum()


# ── Hard detection loss ───────────────────────────────────────────────────────

def hard_det_loss(
    preds,
    targets,
    anchors: torch.Tensor,
) -> tuple:
    """
    Standard single-stage detection loss.
    preds   : list of (cls_map, reg_map) per head level
    targets : list of dicts with 'boxes' key per image
    Returns : (scalar loss, num_positives)
    """
    device = anchors.device
    B = len(targets)

    cls_list, reg_list = [], []
    for (cm, rm) in preds:
        bs = cm.shape[0]
        cls_list.append(cm.permute(0, 2, 3, 1).reshape(bs, -1, 1))
        reg_list.append(rm.permute(0, 2, 3, 1).reshape(bs, -1, 4))
    cls_pred = torch.cat(cls_list, dim=1)   # [B, A, 1]
    reg_pred = torch.cat(reg_list, dim=1)   # [B, A, 4]

    total_cls = total_reg = torch.tensor(0.0, device=device)
    total_pos = 0

    for b in range(B):
        gt = targets[b]["boxes"].to(device)
        matched_idxs, labels = match_anchors(anchors, gt)
        valid = (labels == 1) | (labels == 0)

        total_cls = total_cls + focal_loss(
            cls_pred[b][valid].squeeze(-1), labels[valid])

        pos = labels == 1
        if pos.sum() > 0:
            pred_boxes = decode_boxes(anchors[pos], reg_pred[b][pos])
            total_reg  = total_reg + giou_loss(pred_boxes, gt[matched_idxs[pos]])
            total_pos += pos.sum().item()

    n = max(total_pos, 1)
    return 1.0 * total_cls / n + 2.0 * total_reg / n, total_pos


# ── Knowledge Distillation Loss ───────────────────────────────────────────────

class KDLoss(nn.Module):
    """
    Full KD loss combining three signals:

    L_total = L_hard + w_feat * L_feat + w_soft * L_soft

    L_hard : Focal + GIoU against ground-truth boxes
    L_feat : Cosine similarity between student and teacher feature maps
             (cosine is scale-invariant — robust when teacher/student
              activation magnitudes differ greatly)
    L_soft : KL divergence of classification predictions at temperature T
             (teaches student the teacher's confidence and uncertainty)
    """

    def __init__(
        self,
        w_feat: float   = config.KD_W_FEAT,
        w_soft: float   = config.KD_W_SOFT,
        temperature: float = config.KD_TEMP,
    ):
        super().__init__()
        self.w_feat = w_feat
        self.w_soft = w_soft
        self.T      = temperature

    def feature_loss(
        self,
        s_feats: dict,
        t_feats: dict,
    ) -> torch.Tensor:
        """
        Cosine similarity loss on projected feature maps.
        s_feats: {'feat1': [B,64,18,18], 'feat2': [B,64,9,9]}
        t_feats: {'P3':    [B,64,H,W],   'P4':    [B,64,H,W]}
        """
        total = 0.0
        pairs = [
            (s_feats["feat1"], t_feats["P3"]),
            (s_feats["feat2"], t_feats["P4"]),
        ]
        for sf, tf in pairs:
            if sf.shape[2:] != tf.shape[2:]:
                tf = F.adaptive_avg_pool2d(tf, sf.shape[2:])
            # [B, C, N] — cosine per spatial position
            sf_flat = sf.flatten(2)
            tf_flat = tf.flatten(2).detach()
            cos_sim = F.cosine_similarity(sf_flat, tf_flat, dim=1)  # [B, N]
            total  += (1.0 - cos_sim).mean()
        return torch.tensor(total / 2.0, device=sf.device) \
            if isinstance(total, float) else total / 2.0

    def soft_loss(self, s_preds, t_preds) -> torch.Tensor:
        """KL divergence of classification outputs at temperature T."""
        total = torch.tensor(0.0, device=s_preds[0][0].device)
        n = 0
        for (sc, _), (tc, _) in zip(s_preds, t_preds):
            # Use only face-class channel
            sc = sc[:, :1].flatten(1)               # [B, H*W]
            tc = tc[:, :1].flatten(1).detach()
            if sc.shape[1] != tc.shape[1]:
                tc = F.adaptive_avg_pool1d(
                    tc.unsqueeze(1), sc.shape[1]).squeeze(1)
            s_lsm = F.log_softmax(sc / self.T, dim=1)
            t_sm  = F.softmax(tc   / self.T, dim=1)
            total = total + F.kl_div(s_lsm, t_sm, reduction="batchmean") * (self.T ** 2)
            n += 1
        return total / max(n, 1)

    def forward(
        self,
        s_preds,
        s_feats: dict,
        t_preds,
        t_feats: dict,
        targets,
        anchors: torch.Tensor,
    ) -> tuple:
        l_hard, n_pos = hard_det_loss(s_preds, targets, anchors)
        l_feat        = self.feature_loss(s_feats, t_feats)
        l_soft        = self.soft_loss(s_preds, t_preds)

        total = l_hard + self.w_feat * l_feat + self.w_soft * l_soft

        stats = {
            "loss":   total.item(),
            "hard":   l_hard.item(),
            "feat":   l_feat.item(),
            "soft":   l_soft.item(),
            "n_pos":  n_pos,
        }
        return total, stats
