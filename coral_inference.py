"""
coral_inference.py
------------------
Run face detection on the Coral Edge TPU board.

Copy this file and the compiled model to the Coral board, then:
    python coral_inference.py --model face_detector_int8_edgetpu.tflite --image test.jpg

Requirements on Coral board:
    pip install pycoral tflite-runtime opencv-python-headless numpy
"""

import argparse
import time
import cv2
import numpy as np

# PyCoral imports — only available on Coral board / with Coral SDK installed
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common as coral_common
    CORAL = True
except ImportError:
    # Fallback to plain TFLite (for testing on a regular PC without Coral)
    import tflite_runtime.interpreter as tflite
    CORAL = False
    print("⚠️  PyCoral not found — running with plain TFLite (CPU, slower)")


# ── Config ────────────────────────────────────────────────────────────────────

IMG_SIZE = 74
MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Student anchor config (must match training)
HEAD1_ANCHOR_SIZES  = [6,  10, 16]
HEAD2_ANCHOR_SIZES  = [22, 32, 46]
HEAD1_STRIDE        = IMG_SIZE / 18
HEAD2_STRIDE        = IMG_SIZE / 9

WEIGHTS = (10.0, 10.0, 5.0, 5.0)


# ── Anchor generation ─────────────────────────────────────────────────────────

def generate_anchors():
    all_anchors = []
    for fsize, stride, sizes in [
        (18, HEAD1_STRIDE, HEAD1_ANCHOR_SIZES),
        ( 9, HEAD2_STRIDE, HEAD2_ANCHOR_SIZES),
    ]:
        for sz in sizes:
            for iy in range(fsize):
                for ix in range(fsize):
                    cx = (ix + 0.5) * stride
                    cy = (iy + 0.5) * stride
                    half = sz / 2.0
                    all_anchors.append([cx-half, cy-half, cx+half, cy+half])
    return np.array(all_anchors, dtype=np.float32)

ANCHORS = generate_anchors()


# ── Box decode ────────────────────────────────────────────────────────────────

def decode_boxes(anchors, deltas):
    wa  = anchors[:, 2] - anchors[:, 0]
    ha  = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + 0.5 * wa
    cya = anchors[:, 1] + 0.5 * ha

    cx = deltas[:, 0] / WEIGHTS[0] * wa  + cxa
    cy = deltas[:, 1] / WEIGHTS[1] * ha  + cya
    w  = np.exp(deltas[:, 2] / WEIGHTS[2]) * wa
    h  = np.exp(deltas[:, 3] / WEIGHTS[3]) * ha

    x1 = cx - 0.5 * w; y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w; y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=1)


# ── NMS (pure numpy) ──────────────────────────────────────────────────────────

def nms(boxes, scores, iou_threshold=0.4):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(bgr):
    h0, w0  = bgr.shape[:2]
    scale   = max(h0, w0) / IMG_SIZE
    small   = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = ((rgb - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]  # [1,3,74,74]
    # Convert to INT8 for Coral
    tensor_int8 = (tensor * 128).clip(-128, 127).astype(np.int8)
    return tensor_int8, scale


# ── Postprocessing ────────────────────────────────────────────────────────────

def postprocess(outputs, scale, conf_thresh=0.4, nms_thresh=0.4):
    """
    outputs: list of 4 numpy arrays [cls1, reg1, cls2, reg2]
    Each cls has shape [1, N_ANCHORS, H, W]
    Each reg has shape [1, N_ANCHORS*4, H, W]
    """
    cls_list, reg_list = [], []
    for i in range(0, len(outputs), 2):
        cm, rm = outputs[i], outputs[i+1]
        bs = cm.shape[0]
        n_anch = cm.shape[1]
        cls_list.append(cm.transpose(0,2,3,1).reshape(bs,-1))
        reg_list.append(rm.transpose(0,2,3,1).reshape(bs,-1,4))

    cls_all = np.concatenate(cls_list, axis=1)[0].astype(np.float32) / 128.0
    reg_all = np.concatenate(reg_list, axis=1)[0].astype(np.float32) / 128.0

    scores = 1.0 / (1.0 + np.exp(-cls_all))   # sigmoid
    keep   = scores > conf_thresh

    if not keep.any():
        return np.zeros((0,4)), np.zeros(0)

    boxes  = decode_boxes(ANCHORS[keep], reg_all[keep])
    boxes  = np.clip(boxes, 0, IMG_SIZE)
    scores = scores[keep]

    idxs   = nms(boxes, scores, nms_thresh)
    boxes  = boxes[idxs] * scale
    scores = scores[idxs]
    return boxes, scores


# ── Interpreter setup ─────────────────────────────────────────────────────────

def make_tflite_interpreter(model_path: str):
    if CORAL:
        interp = make_interpreter(model_path)
    else:
        interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_inference(interp, input_tensor: np.ndarray):
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.set_tensor(input_details[0]["index"], input_tensor)
    interp.invoke()

    return [interp.get_tensor(d["index"]) for d in output_details]


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_detections(image, boxes, scores):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x1, max(y1-4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return image


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="Path to _edgetpu.tflite model")
    p.add_argument("--image",   default=None,  help="Run on a single image")
    p.add_argument("--camera",  action="store_true", help="Run on live camera feed")
    p.add_argument("--output",  default="result.jpg")
    p.add_argument("--conf",    type=float, default=0.4)
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    interp = make_tflite_interpreter(args.model)
    backend = "Coral Edge TPU" if CORAL else "TFLite CPU"
    print(f"Running on  : {backend}")

    if args.image:
        # Single image
        bgr = cv2.imread(args.image)
        assert bgr is not None, f"Cannot read: {args.image}"

        t0 = time.perf_counter()
        tensor, scale = preprocess(bgr)
        outputs       = run_inference(interp, tensor)
        boxes, scores = postprocess(outputs, scale, conf_thresh=args.conf)
        elapsed_ms    = (time.perf_counter() - t0) * 1000

        print(f"Detected {len(boxes)} faces in {elapsed_ms:.1f} ms")
        vis = draw_detections(bgr.copy(), boxes, scores)
        cv2.imwrite(args.output, vis)
        print(f"Result saved → {args.output}")

    elif args.camera:
        # Live camera feed
        cap = cv2.VideoCapture(0)
        print("Camera running — press Q to quit")
        fps_history = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            tensor, scale = preprocess(frame)
            outputs       = run_inference(interp, tensor)
            boxes, scores = postprocess(outputs, scale, conf_thresh=args.conf)
            elapsed       = time.perf_counter() - t0

            fps_history.append(1.0 / elapsed)
            if len(fps_history) > 30: fps_history.pop(0)
            fps = sum(fps_history) / len(fps_history)

            vis = draw_detections(frame, boxes, scores)
            cv2.putText(vis, f"FPS: {fps:.1f}  Faces: {len(boxes)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Face Detection — Coral", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
