"""
dataset.py
----------
WIDER FACE dataset with dual-size loading:
  - 640×640  for the RetinaFace teacher
  -  74×74   for the MAX78000/Coral student

Both sizes see the same random augmentation crop,
so the teacher and student features are spatially consistent.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ── Annotation parser ─────────────────────────────────────────────────────────

def parse_wider_annotation(ann_file: str):
    """
    Parse WIDER FACE annotation file.
    Returns list of {'image_path': str, 'bboxes': np.ndarray [N,4] x1y1x2y2}.
    """
    samples = []
    with open(ann_file) as f:
        lines = [l.strip() for l in f]

    i = 0
    while i < len(lines):
        path = lines[i]; i += 1
        if not path:
            continue
        n = int(lines[i]); i += 1
        boxes = []
        for _ in range(max(n, 1)):
            parts = list(map(int, lines[i].split())); i += 1
            if n == 0:
                break
            x1, y1, w, h = parts[0], parts[1], parts[2], parts[3]
            if w > 0 and h > 0:
                boxes.append([x1, y1, x1 + w, y1 + h])
        samples.append({
            "image_path": path,
            "bboxes": np.array(boxes, dtype=np.float32)
                      if boxes else np.zeros((0, 4), dtype=np.float32),
        })
    return samples


# ── Augmentation pipelines ────────────────────────────────────────────────────

def _train_aug():
    return A.Compose([
        A.RandomResizedCrop(
            size=(config.TEACHER_IMG_SIZE, config.TEACHER_IMG_SIZE),
            scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.ToGray(p=0.05),
    ], bbox_params=A.BboxParams(
        "pascal_voc", label_fields=["labels"], min_visibility=0.3))


def _val_aug():
    return A.Compose([
        A.LongestMaxSize(config.TEACHER_IMG_SIZE),
        A.PadIfNeeded(
            min_height=config.TEACHER_IMG_SIZE, min_width=config.TEACHER_IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT, value=0),
    ], bbox_params=A.BboxParams(
        "pascal_voc", label_fields=["labels"], min_visibility=0.1))


# ── Dataset ───────────────────────────────────────────────────────────────────

class WiderFaceDataset(Dataset):
    """
    Returns per item:
        t_img  : float tensor [3, 640, 640]  — for teacher
        s_img  : float tensor [3,  74,  74]  — for student
        target : dict with 'boxes' [N, 4] in 74×74 coordinates
    """

    def __init__(self, root: str, split: str = "train"):
        assert split in ("train", "val")
        self.img_dir  = os.path.join(root, f"WIDER_{split}", "images")
        ann_file      = os.path.join(
            root, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")
        self.samples  = parse_wider_annotation(ann_file)
        self.aug      = _train_aug() if split == "train" else _val_aug()
        self.norm     = A.Normalize(MEAN, STD)
        self.to_tensor = ToTensorV2()
        self.scale    = config.STUDENT_IMG_SIZE / config.TEACHER_IMG_SIZE

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s   = self.samples[idx]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, s["image_path"])),
            cv2.COLOR_BGR2RGB)

        h, w   = img.shape[:2]
        boxes  = s["bboxes"].copy()
        if len(boxes):
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
            ok = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes = boxes[ok]

        aug = self.aug(
            image=img,
            bboxes=boxes.tolist() if len(boxes) else [],
            labels=[0] * len(boxes))

        large     = aug["image"]                               # 640×640 uint8
        aug_boxes = np.array(aug["bboxes"], dtype=np.float32) \
                    if aug["bboxes"] else np.zeros((0, 4), dtype=np.float32)

        # Teacher image — 640×640 normalised
        t_img = self.to_tensor(image=self.norm(image=large)["image"])["image"]

        # Student image — resize 640→74
        small = cv2.resize(
            large,
            (config.STUDENT_IMG_SIZE, config.STUDENT_IMG_SIZE),
            interpolation=cv2.INTER_LINEAR)
        s_img = self.to_tensor(image=self.norm(image=small)["image"])["image"]

        # Scale boxes to 74×74 coordinate space
        s_boxes = aug_boxes * self.scale
        s_boxes = np.clip(s_boxes, 0, config.STUDENT_IMG_SIZE)

        target = {
            "boxes":    torch.tensor(s_boxes, dtype=torch.float32),
            "image_id": torch.tensor([idx]),
        }
        return t_img, s_img, target


def collate_fn(batch):
    t_imgs, s_imgs, targets = zip(*batch)
    return list(t_imgs), list(s_imgs), list(targets)


def build_loaders(root: str):
    train_ds = WiderFaceDataset(root, split="train")
    val_ds   = WiderFaceDataset(root, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, train_ds, val_ds
