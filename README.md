# Face Detection for Coral Edge TPU / MAX78000

A lightweight face detector trained via **Knowledge Distillation** from RetinaFace (ResNet-50 teacher) to a compact student model optimised for the **Coral Edge TPU** and **MAX78000** microcontroller.

Pipeline: `WIDER FACE dataset` → `KD training (FP32 + QAT)` → `ONNX` → `TFLite INT8` → `Edge TPU .tflite`

---

## Setup

### 1. Clone the repo

```bash
git clone git@github-vatsal:VatsalDixit/Face-detection-Coral-MAX78000.git
cd Face-detection-Coral-MAX78000
```

### 2. Create and activate a conda environment

```bash
conda create -n vatsal_fd python=3.10 -y
conda activate vatsal_fd
```

### 3. Install PyTorch

Install via pip with the CUDA 12.8 index (works on CUDA 12.8+ and 13.x drivers, required for RTX 50xx / Blackwell GPUs):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> For older GPUs (sm_90 and below), you can use the conda channel instead:
> ```bash
> conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
> ```

### 4. Install remaining dependencies

```bash
pip install albumentations opencv-python tqdm tensorboard numpy gdown onnx
```

> **Note:** `tensorflow` and `onnx-tf` are only needed for the Coral export step:
> ```bash
> pip install tensorflow onnx-tf
> ```

---

## Dataset

Download WIDER FACE and the pretrained RetinaFace teacher weights automatically:

```bash
python download_dataset.py
```

This downloads (~1.8 GB total) and extracts:

| Path | Contents |
|------|----------|
| `data/wider_face/WIDER_train/` | Training images |
| `data/wider_face/WIDER_val/` | Validation images |
| `data/wider_face/wider_face_split/` | Annotations |
| `models/retinaface_resnet50.pth` | Teacher weights |

Custom save directory:
```bash
python download_dataset.py --save_dir ./data/wider_face --weights_dir ./models
```

---

## Training

### Standard run (uses `config.py` defaults)

```bash
python train.py
```

### Common options

```bash
# Custom output directory and batch size
python train.py --output_dir runs/exp2 --batch_size 16

# Resume from a checkpoint
python train.py --resume runs/exp1/last.pth

# Full options
python train.py \
  --data_root    ./data/wider_face \
  --output_dir   runs/exp1 \
  --teacher_weights ./models/retinaface_resnet50.pth \
  --batch_size   8 \
  --epochs_fp32  30 \
  --epochs_qat   20 \
  --lr           1e-3
```

Training runs **30 epochs FP32** then automatically switches to **20 epochs QAT** (INT8 simulation). Checkpoints are saved to `runs/exp1/`:

| File | Description |
|------|-------------|
| `last.pth` | Most recent epoch |
| `best.pth` | Best validation loss |
| `epoch_XXX.pth` | Every 5 epochs |
| `train.log` | Full training log |
| `tb/` | TensorBoard logs |

### Monitor training with TensorBoard

```bash
tensorboard --logdir runs/
```

---

## Configuration

All hyperparameters are in [config.py](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 8 | Increase to 16 with ≥16 GB VRAM |
| `EPOCHS_FP32` | 30 | Float32 training epochs |
| `EPOCHS_QAT` | 20 | Quantization-aware training epochs |
| `LR` | 1e-3 | Initial learning rate |
| `STUDENT_IMG_SIZE` | 74 | Input resolution (74×74, Coral-compatible) |
| `KD_TEMP` | 4.0 | Knowledge distillation temperature |

---

## Export to Coral Edge TPU

After training completes, convert the checkpoint through the full export pipeline:

```bash
python export_coral.py
```

Or with explicit paths:

```bash
python export_coral.py \
  --checkpoint runs/exp1/best.pth \
  --output_dir exports/ \
  --data_root  ./data/wider_face
```

This produces:

| File | Description |
|------|-------------|
| `exports/face_detector.onnx` | ONNX model |
| `exports/face_detector_tf/` | TF SavedModel |
| `exports/face_detector_int8.tflite` | TFLite INT8 model |
| `exports/sample_input.npy` | Sample input for verification |

### Compile for Coral (Linux only)

```bash
# Install Edge TPU compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update && sudo apt install edgetpu-compiler

# Compile
edgetpu_compiler exports/face_detector_int8.tflite
# Output: exports/face_detector_int8_edgetpu.tflite
```

---

## Inference on Coral Board

Copy the compiled model and inference script to the board:

```bash
scp exports/face_detector_int8_edgetpu.tflite mendel@<CORAL_IP>:~/
scp coral_inference.py mendel@<CORAL_IP>:~/
```

Install dependencies on the board:

```bash
pip install pycoral tflite-runtime opencv-python-headless numpy
```

Run inference:

```bash
# Single image
python coral_inference.py \
  --model face_detector_int8_edgetpu.tflite \
  --image test.jpg \
  --output result.jpg

# Live camera feed
python coral_inference.py \
  --model face_detector_int8_edgetpu.tflite \
  --camera

# Adjust confidence threshold
python coral_inference.py \
  --model face_detector_int8_edgetpu.tflite \
  --image test.jpg \
  --conf 0.5
```

> Without a Coral board, the script falls back to TFLite CPU automatically.

---

## Project Structure

```
.
├── config.py              # All hyperparameters
├── train.py               # Main training script
├── download_dataset.py    # WIDER FACE + teacher weights downloader
├── export_coral.py        # PyTorch → ONNX → TFLite → Edge TPU pipeline
├── coral_inference.py     # Inference script for Coral board
├── dataset.py             # Dataset and dataloader
├── models/
│   ├── teacher.py         # RetinaFace teacher (frozen)
│   └── student.py         # Lightweight student detector
├── anchors.py             # Anchor generation
├── losses.py              # Knowledge distillation loss
├── requirements.txt
├── data/wider_face/       # Downloaded dataset (gitignored)
├── models/                # Downloaded weights (gitignored)
├── runs/                  # Training outputs (gitignored)
└── exports/               # Export outputs (gitignored)
```
