"""
export_coral.py
---------------
Converts the trained PyTorch student checkpoint to Coral Edge TPU format.

Pipeline:
    PyTorch QAT checkpoint (.pth)
        ↓  convert QAT → INT8
        ↓  torch.onnx.export
    ONNX model (.onnx)
        ↓  onnx-tf backend
    TF SavedModel
        ↓  TFLiteConverter + representative dataset (INT8 calibration)
    TFLite INT8 model (.tflite)
        ↓  edgetpu_compiler  [run on Coral machine or Linux PC]
    Coral-compiled model (_edgetpu.tflite)
        ↓  copy to Coral board
    Run with PyCoral

Usage:
    python export_coral.py
    python export_coral.py --checkpoint runs/exp1/best.pth --output_dir exports/
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

import config
from models.student import StudentDetector
from dataset import parse_wider_annotation


MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD_NP  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Load checkpoint ───────────────────────────────────────────────────────────

def load_student(checkpoint_path: str) -> StudentDetector:
    model = StudentDetector()
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"✅  Loaded checkpoint: epoch {ckpt.get('epoch', '?')+1}, "
          f"val_loss={ckpt.get('best_val', float('nan')):.4f}")
    print(f"    QAT was active: {ckpt.get('qat', False)}")
    return model


# ── Step 1: Convert QAT → INT8 ────────────────────────────────────────────────

def convert_to_int8(model: StudentDetector) -> StudentDetector:
    """Convert QAT model to a true INT8 PyTorch model."""
    model_int8 = torch.quantization.convert(model, inplace=False)
    model_int8.eval()
    print("✅  Converted to INT8 (PyTorch quantized model)")
    return model_int8


# ── Step 2: Export ONNX ───────────────────────────────────────────────────────

def export_onnx(model: StudentDetector, output_path: str):
    """Export FP32 model to ONNX (onnx-tf works better with FP32 input)."""
    dummy = torch.zeros(1, 3, config.STUDENT_IMG_SIZE, config.STUDENT_IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=11,
        input_names=["input"],
        output_names=["cls_head1", "reg_head1", "cls_head2", "reg_head2"],
        dynamic_axes={"input": {0: "batch_size"}},
        verbose=False,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"✅  ONNX exported → {output_path}  ({size_mb:.1f} MB)")


# ── Step 3: ONNX → TF SavedModel ─────────────────────────────────────────────

def onnx_to_tf(onnx_path: str, tf_dir: str):
    """Convert ONNX model to TensorFlow SavedModel."""
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        print("Installing onnx-tf ...")
        os.system(f"{sys.executable} -m pip install onnx onnx-tf")
        import onnx
        from onnx_tf.backend import prepare

    print("Converting ONNX → TF SavedModel ...")
    onnx_model = onnx.load(onnx_path)
    tf_rep     = prepare(onnx_model)
    tf_rep.export_graph(tf_dir)
    print(f"✅  TF SavedModel → {tf_dir}")


# ── Step 4: TF SavedModel → TFLite INT8 ──────────────────────────────────────

def make_representative_dataset(data_root: str):
    """
    Generator that yields sample inputs for INT8 calibration.
    TFLite converter uses these to determine the quantization ranges
    for each layer's activations.
    """
    val_ann = os.path.join(
        data_root, "wider_face_split", "wider_face_val_bbx_gt.txt")
    val_img = os.path.join(data_root, "WIDER_val", "images")
    samples = parse_wider_annotation(val_ann)[:200]   # 200 images is enough

    def generator():
        for s in samples:
            bgr = cv2.imread(os.path.join(val_img, s["image_path"]))
            if bgr is None:
                continue
            small = cv2.resize(
                bgr,
                (config.STUDENT_IMG_SIZE, config.STUDENT_IMG_SIZE),
                interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = ((rgb - MEAN_NP) / STD_NP).transpose(2, 0, 1)[np.newaxis]  # [1,3,74,74]
            yield [tensor]

    return generator


def tf_to_tflite_int8(tf_dir: str, tflite_path: str, data_root: str):
    """Convert TF SavedModel to TFLite INT8 with full integer quantization."""
    try:
        import tensorflow as tf
    except ImportError:
        print("Installing tensorflow ...")
        os.system(f"{sys.executable} -m pip install tensorflow")
        import tensorflow as tf

    print("Converting TF SavedModel → TFLite INT8 ...")

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)
    converter.optimizations                = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset       = make_representative_dataset(data_root)
    converter.target_spec.supported_ops    = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type         = tf.int8
    converter.inference_output_type        = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1e3
    print(f"✅  TFLite INT8 → {tflite_path}  ({size_kb:.0f} KB)")


# ── Step 5: Save sample input ─────────────────────────────────────────────────

def save_sample_input(data_root: str, output_path: str):
    """Save a representative input as .npy for verification."""
    val_ann = os.path.join(
        data_root, "wider_face_split", "wider_face_val_bbx_gt.txt")
    val_img = os.path.join(data_root, "WIDER_val", "images")
    sample_path = parse_wider_annotation(val_ann)[0]["image_path"]

    bgr   = cv2.imread(os.path.join(val_img, sample_path))
    small = cv2.resize(bgr, (config.STUDENT_IMG_SIZE, config.STUDENT_IMG_SIZE))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr   = ((rgb - MEAN_NP) / STD_NP).transpose(2, 0, 1)[np.newaxis]

    np.save(output_path, arr)
    print(f"✅  Sample input .npy → {output_path}  shape={arr.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Export student model for Coral Edge TPU")
    p.add_argument("--checkpoint",  default="runs/exp1/best.pth")
    p.add_argument("--output_dir",  default=config.EXPORT_DIR)
    p.add_argument("--data_root",   default=config.DATA_ROOT)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    onnx_path    = os.path.join(args.output_dir, "face_detector.onnx")
    tf_dir       = os.path.join(args.output_dir, "face_detector_tf")
    tflite_path  = os.path.join(args.output_dir, "face_detector_int8.tflite")
    sample_path  = os.path.join(args.output_dir, "sample_input.npy")

    print("\n" + "=" * 55)
    print("  Coral Edge TPU Export Pipeline")
    print("=" * 55 + "\n")

    # Load FP32 model (not INT8 — onnx-tf works better from FP32)
    model = load_student(args.checkpoint)
    model.eval()

    # Step 1: ONNX
    export_onnx(model, onnx_path)

    # Step 2: TF SavedModel
    onnx_to_tf(onnx_path, tf_dir)

    # Step 3: TFLite INT8
    tf_to_tflite_int8(tf_dir, tflite_path, args.data_root)

    # Step 4: Sample input
    save_sample_input(args.data_root, sample_path)

    # Done
    print("\n" + "=" * 55)
    print("  Export complete. Files:")
    for f in [onnx_path, tflite_path, sample_path]:
        kb = os.path.getsize(f) / 1e3
        print(f"  📄  {f}  ({kb:.0f} KB)")
    print("=" * 55)
    print("""
  ─── Next steps (run on a Linux machine with Coral SDK) ───

  1. Install Edge TPU compiler:
     curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
     echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \\
       | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
     sudo apt update && sudo apt install edgetpu-compiler

  2. Compile for Coral:
     edgetpu_compiler exports/face_detector_int8.tflite
     # → generates: face_detector_int8_edgetpu.tflite

  3. Copy to Coral board:
     scp face_detector_int8_edgetpu.tflite  mendel@<CORAL_IP>:~/
     scp sample_input.npy                  mendel@<CORAL_IP>:~/

  4. Run inference on Coral:
     python coral_inference.py --model face_detector_int8_edgetpu.tflite

  See coral_inference.py for the full inference script.
""")


if __name__ == "__main__":
    main()
