"""
train.py
--------
Main training script: RetinaFace (teacher) → Student (KD) → QAT

Usage:
    python train.py                          # uses config.py defaults
    python train.py --resume runs/exp1/last.pth
    python train.py --output_dir runs/exp2 --batch_size 16
"""

import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from models          import load_teacher, StudentDetector
from dataset         import build_loaders
from anchors         import generate_student_anchors
from losses          import KDLoss


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ],
    )
    return logging.getLogger("train")


# ── LR schedule: linear warmup + cosine decay ────────────────────────────────

def build_scheduler(optimizer, total_epochs: int):
    def lr_lambda(ep):
        if ep < config.WARMUP_EPOCHS:
            return (ep + 1) / config.WARMUP_EPOCHS
        t = (ep - config.WARMUP_EPOCHS) / max(1, total_epochs - config.WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_ckpt(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_ckpt(path: str, student, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    student.load_state_dict(ckpt["model"])
    if optimizer   and "optimizer"  in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler   and "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("qat", False)


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(
    student, teacher, loader, criterion,
    anchors, optimizer, device, is_train: bool
):
    student.train(is_train)
    teacher.eval()

    totals = {"loss": 0.0, "hard": 0.0, "feat": 0.0, "soft": 0.0, "n_pos": 0}
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="train" if is_train else "val ", leave=False)
        for t_imgs, s_imgs, targets in pbar:
            t_batch = torch.stack(list(t_imgs)).to(device)
            s_batch = torch.stack(list(s_imgs)).to(device)

            # Teacher forward (frozen — no grad even inside enable_grad context)
            with torch.no_grad():
                t_preds, t_feats = teacher(t_batch, return_features=True)

            # Student forward
            s_preds, s_feats = student(s_batch, return_features=True)

            # KD loss
            loss, stats = criterion(
                s_preds, s_feats, t_preds, t_feats, list(targets), anchors)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                optimizer.step()

            for k in ("loss", "hard", "feat", "soft"):
                totals[k] += stats[k]
            totals["n_pos"] += stats["n_pos"]
            n += 1

            pbar.set_postfix(loss=f"{stats['loss']:.4f}", pos=stats["n_pos"])

    return {k: v / max(n, 1) for k, v in totals.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RetinaFace → Student KD training")
    p.add_argument("--data_root",    default=config.DATA_ROOT)
    p.add_argument("--output_dir",   default=config.OUTPUT_DIR)
    p.add_argument("--teacher_weights", default="./models/retinaface_resnet50.pth")
    p.add_argument("--batch_size",   type=int,   default=config.BATCH_SIZE)
    p.add_argument("--epochs_fp32",  type=int,   default=config.EPOCHS_FP32)
    p.add_argument("--epochs_qat",   type=int,   default=config.EPOCHS_QAT)
    p.add_argument("--lr",           type=float, default=config.LR)
    p.add_argument("--resume",       default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    total_epochs = args.epochs_fp32 + args.epochs_qat

    logger.info(f"Device       : {device}")
    if device.type == "cuda":
        logger.info(f"GPU          : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"Batch size   : {args.batch_size}")
    logger.info(f"Epochs FP32  : {args.epochs_fp32}")
    logger.info(f"Epochs QAT   : {args.epochs_qat}")
    logger.info(f"Output dir   : {args.output_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Building dataloaders ...")
    # Override batch size from args
    config.BATCH_SIZE = args.batch_size
    train_loader, val_loader, train_ds, val_ds = build_loaders(args.data_root)
    logger.info(f"  Train: {len(train_ds):,} images  |  Val: {len(val_ds):,} images")

    # ── Models ────────────────────────────────────────────────────────────────
    logger.info("Loading teacher (RetinaFace ResNet-50, frozen) ...")
    teacher = load_teacher(args.teacher_weights, device)
    n_t = sum(p.numel() for p in teacher.parameters())
    logger.info(f"  Teacher params: {n_t/1e6:.1f}M")

    logger.info("Building student ...")
    student = StudentDetector().to(device)
    n_s = sum(p.numel() for p in student.parameters())
    logger.info(f"  Student params: {n_s/1e3:.0f}K  (~{n_s/1024:.0f} KB INT8)")

    # ── Training setup ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, total_epochs)
    criterion = KDLoss()
    anchors   = generate_student_anchors(device)

    start_epoch = 0
    best_val    = float("inf")
    qat_active  = False

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, best_val, qat_active = load_ckpt(
            args.resume, student, optimizer, scheduler)
        start_epoch += 1
        if qat_active:
            logger.info("  QAT was active in checkpoint — re-enabling ...")
            student.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(student, inplace=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("  Starting training")
    logger.info("=" * 55)

    for ep in range(start_epoch, total_epochs):

        # Switch to QAT after FP32 warmup
        if ep == args.epochs_fp32 and not qat_active:
            logger.info(f"\n  ─── Enabling QAT at epoch {ep + 1} ───")
            student.train()
            student.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(student, inplace=True)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1
            qat_active = True
            logger.info("  ✅ QAT active — INT8 weights simulated during forward pass")

        t0    = time.time()
        phase = "QAT " if qat_active else "FP32"

        tr = run_epoch(student, teacher, train_loader, criterion,
                       anchors, optimizer, device, is_train=True)
        vl = run_epoch(student, teacher, val_loader,   criterion,
                       anchors, optimizer, device, is_train=False)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        is_best = vl["loss"] < best_val
        if is_best:
            best_val = vl["loss"]

        logger.info(
            f"[{phase}] Ep {ep+1:03d}/{total_epochs}  "
            f"train={tr['loss']:.4f} (hard={tr['hard']:.3f} feat={tr['feat']:.3f} "
            f"soft={tr['soft']:.3f} pos={int(tr['n_pos'])})  "
            f"val={vl['loss']:.4f}  lr={lr:.2e}  {time.time()-t0:.0f}s"
            + ("  ⭐ best" if is_best else "")
        )

        # TensorBoard
        for tag, stats in [("train", tr), ("val", vl)]:
            writer.add_scalar(f"{tag}/loss_total", stats["loss"], ep)
            writer.add_scalar(f"{tag}/loss_hard",  stats["hard"], ep)
            writer.add_scalar(f"{tag}/loss_feat",  stats["feat"], ep)
            writer.add_scalar(f"{tag}/loss_soft",  stats["soft"], ep)
        writer.add_scalar("lr", lr, ep)

        # Checkpointing
        ckpt = {
            "epoch":     ep,
            "model":     student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val":  best_val,
            "qat":       qat_active,
        }
        save_ckpt(ckpt, os.path.join(args.output_dir, "last.pth"))
        if is_best:
            save_ckpt(ckpt, os.path.join(args.output_dir, "best.pth"))
            logger.info(f"  ✅ Saved best checkpoint → {args.output_dir}/best.pth")
        if (ep + 1) % config.SAVE_EVERY == 0:
            save_ckpt(ckpt, os.path.join(args.output_dir, f"epoch_{ep+1:03d}.pth"))

    writer.close()
    logger.info(f"\n✅ Training complete — best val loss: {best_val:.4f}")
    logger.info(f"   Best checkpoint : {args.output_dir}/best.pth")
    logger.info(f"   Run export      : python export_coral.py")


if __name__ == "__main__":
    main()
