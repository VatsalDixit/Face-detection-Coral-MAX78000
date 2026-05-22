"""
download_dataset.py
-------------------
Downloads the official WIDER FACE dataset from Google Drive and
the annotation file from the official site.

Official source: http://shuoyang1213.me/WIDERFACE/

Usage:
    python download_dataset.py
    python download_dataset.py --save_dir ./data/wider_face
"""

import os
import sys
import zipfile
import argparse
import urllib.request

try:
    import gdown
except ImportError:
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown


# ── Official file registry ────────────────────────────────────────────────────
# Google Drive IDs from http://shuoyang1213.me/WIDERFACE/
GDRIVE_FILES = {
    "WIDER_train.zip": "15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M",   # ~1.4 GB
    "WIDER_val.zip":   "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q",   # ~360 MB
}

# Annotation file hosted directly on official site
ANN_URL = (
    "http://shuoyang1213.me/WIDERFACE/support/"
    "bbx_annotation/wider_face_split.zip"
)

# Pretrained RetinaFace ResNet-50 weights
# Source: biubug6/Pytorch_Retinaface — trained on WIDER FACE (96.9% AP Easy)
RETINAFACE_WEIGHTS_ID = "14KX6VqF69MdSPk3Tr9PlDYbq7ArpdNUW"


def extract(zip_path, dest_dir, keep=False):
    print(f"  Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    if not keep:
        os.remove(zip_path)
    print("  Done.")


def download_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1024 / 1024
        print(f"\r  {pct:5.1f}%  ({mb:.1f} MB)", end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./data/wider_face")
    parser.add_argument("--weights_dir", default="./models")
    parser.add_argument("--keep_zips", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.weights_dir, exist_ok=True)

    print("\n" + "=" * 52)
    print("  WIDER FACE Dataset Downloader")
    print(f"  Save dir : {args.save_dir}")
    print("=" * 52 + "\n")

    # ── Images ────────────────────────────────────────────────────────────────
    for fname, fid in GDRIVE_FILES.items():
        dest   = os.path.join(args.save_dir, fname)
        folder = os.path.join(args.save_dir, fname.replace(".zip", ""))
        if os.path.exists(folder):
            print(f"✅  {fname} already extracted")
            continue
        print(f"📥  Downloading {fname} from Google Drive ...")
        download_gdrive(fid, dest)
        extract(dest, args.save_dir, keep=args.keep_zips)

    # ── Annotations ───────────────────────────────────────────────────────────
    ann_dir = os.path.join(args.save_dir, "wider_face_split")
    if os.path.exists(ann_dir):
        print("✅  Annotations already extracted")
    else:
        print("📥  Downloading annotations from shuoyang1213.me ...")
        ann_zip = os.path.join(args.save_dir, "wider_face_split.zip")
        urllib.request.urlretrieve(ANN_URL, ann_zip, reporthook=progress_hook)
        print()
        extract(ann_zip, args.save_dir, keep=args.keep_zips)

    # ── RetinaFace pretrained weights ─────────────────────────────────────────
    weights_path = os.path.join(args.weights_dir, "retinaface_resnet50.pth")
    if os.path.exists(weights_path):
        print("✅  RetinaFace pretrained weights already downloaded")
    else:
        print("📥  Downloading RetinaFace ResNet-50 pretrained weights ...")
        download_gdrive(RETINAFACE_WEIGHTS_ID, weights_path)

    # ── Verify ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Verification:")
    checks = {
        "Train images" : os.path.join(args.save_dir, "WIDER_train", "images"),
        "Val images"   : os.path.join(args.save_dir, "WIDER_val",   "images"),
        "Train annots" : os.path.join(args.save_dir, "wider_face_split",
                                      "wider_face_train_bbx_gt.txt"),
        "Val annots"   : os.path.join(args.save_dir, "wider_face_split",
                                      "wider_face_val_bbx_gt.txt"),
        "Teacher weights": weights_path,
    }
    all_ok = True
    for name, path in checks.items():
        ok = os.path.exists(path)
        print(f"  {'✅' if ok else '❌ MISSING'}  {name}")
        if not ok:
            all_ok = False

    print("=" * 52)
    if all_ok:
        print("\n  ✅ All files ready. Run training with:")
        print("     python train.py\n")
    else:
        print("\n  ❌ Some files missing. Re-run this script.\n")


if __name__ == "__main__":
    main()
