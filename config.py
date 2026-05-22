"""
config.py — All hyperparameters in one place.
Edit this file before training; do not scatter magic numbers across scripts.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = "./data/wider_face"        # where WIDER FACE is stored
OUTPUT_DIR   = "./runs/exp1"              # checkpoints + tensorboard logs
EXPORT_DIR   = "./exports"               # ONNX, TFLite outputs

# ── Dataset ───────────────────────────────────────────────────────────────────
TEACHER_IMG_SIZE = 640     # teacher sees full-resolution crops
STUDENT_IMG_SIZE = 74      # student sees 74×74 (Coral-compatible)
NUM_WORKERS      = 4       # dataloader workers (reduce to 2 if RAM is tight)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 8           # increase to 16 if GPU has ≥ 16 GB VRAM
EPOCHS_FP32  = 30          # standard float32 training
EPOCHS_QAT   = 20          # quantization-aware training (INT8 simulation)
LR           = 1e-3        # initial learning rate
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3          # linear LR warmup at the start

# ── Knowledge Distillation ────────────────────────────────────────────────────
KD_TEMP      = 4.0         # temperature for soft label distillation
KD_W_FEAT    = 0.5         # weight on feature cosine loss
KD_W_SOFT    = 0.4         # weight on soft label KL loss
# Total loss = L_hard + KD_W_FEAT * L_feat + KD_W_SOFT * L_soft

# ── Anchor matching ───────────────────────────────────────────────────────────
POS_IOU_THRESH = 0.4       # anchor is positive if IoU with any GT >= this
NEG_IOU_THRESH = 0.2       # anchor is background if IoU < this

# ── Student anchor config ─────────────────────────────────────────────────────
# Two detection heads in student, at 18×18 and 9×9 feature maps
HEAD1_ANCHOR_SIZES  = [6,  10, 16]    # px in 74×74 image space
HEAD2_ANCHOR_SIZES  = [22, 32, 46]
HEAD1_STRIDE = STUDENT_IMG_SIZE / 18  # ≈4.1
HEAD2_STRIDE = STUDENT_IMG_SIZE / 9   # ≈8.2

# ── Teacher anchor config ─────────────────────────────────────────────────────
TEACHER_ANCHOR_CFG = {
    "P2": {"sizes": [16,  32],   "stride": 8},
    "P3": {"sizes": [64,  128],  "stride": 16},
    "P4": {"sizes": [256, 512],  "stride": 32},
    "P5": {"sizes": [512, 1024], "stride": 64},
}

# ── Inference ─────────────────────────────────────────────────────────────────
CONF_THRESH  = 0.4         # confidence threshold at inference
NMS_THRESH   = 0.4         # NMS IoU threshold

# ── Logging ───────────────────────────────────────────────────────────────────
SAVE_EVERY   = 5           # save a numbered checkpoint every N epochs
LOG_INTERVAL = 50          # print loss every N batches
