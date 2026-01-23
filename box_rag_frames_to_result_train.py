# -*- coding: utf-8 -*-
"""
Box existing selected frames (jpg) under RAG_frames_selected and save to Result_train.

Input structure example:
  /.../RAG_frames_selected/
      S011/
        arytenoids/
          frames_s3_n16/
            000472.jpg ...
          frames_s3_n16.json
        epiglottis/
          frames_s3_n16/
            ....

Output structure (mirrors input):
  /.../Result_train/
      S011/
        arytenoids/
          frames_s3_n16/
            000472.jpg ...
          frames_s3_n16.json   (copied)
        epiglottis/...

Boxing method:
  - CNN detector -> filter by class -> NMS -> top score -> pad -> draw rectangle
  - if no detection: fallback to whole-image box

Run:
  python box_rag_frames_to_result_train.py
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import torch
import numpy as np
from torchvision.ops import nms

from config import CLASSES, DEVICE
from model import create_model


# ===================== HARD-CODED PATHS (edit if needed) =====================
IN_ROOT = "/research/home/he234993/platypus/all_data/RAG_frames_selected"
OUT_ROOT = "/research/home/he234993/platypus/Result_train"
WEIGHT_PATH = "/research/home/he234993/last_model.pth"
# =============================================================================


# -------------------- DETECTOR CONFIG --------------------
SCORE_THR = 0.15
SCORE_THR_FALLBACK = 0.05
NMS_IOU = 0.5
PAD_RATIO = 0.15
# ---------------------------------------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_target_class_id(classes: List[str], target_name: str) -> Optional[int]:
    """Exact match first, then substring match."""
    t = (target_name or "").lower().strip()
    for i, n in enumerate(classes):
        if (n or "").lower().strip() == t:
            return i
    hits = [i for i, n in enumerate(classes) if t in (n or "").lower()]
    return hits[0] if hits else None


def find_epiglottis_class_id(classes: List[str]) -> Optional[int]:
    """Try common substrings."""
    for key in ["epiglott", "epiglot"]:
        hits = [i for i, n in enumerate(classes) if key in (n or "").lower()]
        if hits:
            return hits[0]
    return None


def load_detector(weight_path: str):
    assert os.path.isfile(weight_path), f"Weight not found: {weight_path}"
    num_classes = len(CLASSES)
    model = create_model(num_classes=num_classes)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def detect_one_box(
    model,
    bgr_img: np.ndarray,
    target_id: int,
    score_thr: float,
    nms_iou: float,
    pad_ratio: float,
) -> Tuple[List[int], float]:
    """
    Return (box, score). Raise ValueError if no detection.
    box = [x1,y1,x2,y2]
    """
    H, W = bgr_img.shape[:2]

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    out = model(tensor)[0]
    boxes = out["boxes"].detach().cpu()
    labels = out["labels"].detach().cpu()
    scores = out["scores"].detach().cpu()

    keep = scores >= float(score_thr)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    cls_keep = labels == int(target_id)
    boxes, scores = boxes[cls_keep], scores[cls_keep]

    if len(boxes) == 0:
        raise ValueError("no_det")

    keep_idx = nms(boxes, scores, iou_threshold=float(nms_iou))
    boxes, scores = boxes[keep_idx], scores[keep_idx]

    top = torch.argmax(scores)
    bb = boxes[top].to(torch.int).tolist()
    sc = float(scores[top])

    x1, y1, x2, y2 = bb
    w, h = max(1, x2 - x1), max(1, y2 - y1)

    pad = float(pad_ratio)
    x1 = max(0, int(x1 - pad * w))
    y1 = max(0, int(y1 - pad * h))
    x2 = min(W - 1, int(x2 + pad * w))
    y2 = min(H - 1, int(y2 + pad * h))

    return [x1, y1, x2, y2], sc


def draw_box(bgr_img: np.ndarray, box: List[int], label: str, score: float) -> np.ndarray:
    vis = bgr_img.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        vis,
        f"{label} {score:.2f}",
        (x1, max(0, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def iter_frames_dirs(in_root: Path):
    """
    Yield tuples:
      (sample_id_dir, organ_dir, frames_dir)
    where frames_dir contains jpgs.
    """
    for sample_dir in sorted(in_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        for organ in ["arytenoids", "epiglottis"]:
            organ_dir = sample_dir / organ
            if not organ_dir.is_dir():
                continue
            for frames_dir in sorted(organ_dir.iterdir()):
                if frames_dir.is_dir() and frames_dir.name.startswith("frames_"):
                    yield sample_dir.name, organ, frames_dir


def copy_json_sidecar(sample_id: str, organ: str, frames_dirname: str, in_root: Path, out_root: Path):
    """
    If there is a json file like:
      IN_ROOT/S011/arytenoids/frames_s3_n16.json
    copy it to:
      OUT_ROOT/S011/arytenoids/frames_s3_n16.json
    """
    src = in_root / sample_id / organ / f"{frames_dirname}.json"
    if src.is_file():
        dst = out_root / sample_id / organ / f"{frames_dirname}.json"
        ensure_dir(str(dst.parent))
        try:
            shutil.copy2(str(src), str(dst))
        except Exception as e:
            print(f"[WARN] copy json failed: {src} -> {dst} err={e}")


def main():
    in_root = Path(IN_ROOT)
    out_root = Path(OUT_ROOT)
    assert in_root.is_dir(), f"IN_ROOT not found: {IN_ROOT}"
    ensure_dir(str(out_root))

    print("[INFO] IN_ROOT    =", str(in_root))
    print("[INFO] OUT_ROOT   =", str(out_root))
    print("[INFO] WEIGHT     =", WEIGHT_PATH)

    ary_id = find_target_class_id(CLASSES, "arytenoid")
    epi_id = find_epiglottis_class_id(CLASSES)

    if ary_id is None or epi_id is None:
        print("[FATAL] Cannot find class id in CLASSES.")
        print("CLASSES =", list(enumerate(CLASSES)))
        raise SystemExit(1)

    print(f"[INFO] ary_id={ary_id} class={CLASSES[ary_id]}")
    print(f"[INFO] epi_id={epi_id} class={CLASSES[epi_id]}")

    model = load_detector(WEIGHT_PATH)

    total_imgs = 0
    total_dirs = 0

    for sample_id, organ, frames_dir in iter_frames_dirs(in_root):
        total_dirs += 1
        frames_dirname = frames_dir.name

        # output dir mirrors input dir
        out_dir = out_root / sample_id / organ / frames_dirname
        ensure_dir(str(out_dir))

        # also copy sidecar json if exists
        copy_json_sidecar(sample_id, organ, frames_dirname, in_root, out_root)

        # choose target class by organ
        if organ == "arytenoids":
            target_id = int(ary_id)
            target_label = CLASSES[ary_id]
        else:
            target_id = int(epi_id)
            target_label = CLASSES[epi_id]

        jpgs = sorted(list(frames_dir.glob("*.jpg")))
        if not jpgs:
            print(f"[WARN] no jpgs: {frames_dir}")
            continue

        for img_path in jpgs:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[WARN] imread failed: {img_path}")
                continue

            # detect (two thresholds) + fallback to whole image
            try:
                box, sc = detect_one_box(model, bgr, target_id, SCORE_THR, NMS_IOU, PAD_RATIO)
            except Exception:
                try:
                    box, sc = detect_one_box(model, bgr, target_id, SCORE_THR_FALLBACK, NMS_IOU, PAD_RATIO)
                except Exception:
                    H, W = bgr.shape[:2]
                    box, sc = [0, 0, W - 1, H - 1], 0.0

            vis = draw_box(bgr, box, target_label, sc)

            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), vis)
            total_imgs += 1

        print(f"[DONE] {sample_id}/{organ}/{frames_dirname} -> {len(jpgs)} imgs")

    print(f"[ALL DONE] dirs={total_dirs}, images={total_imgs}")
    print(f"[OUT] {OUT_ROOT}")


if __name__ == "__main__":
    main()
