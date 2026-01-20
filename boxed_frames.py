# -*- coding: utf-8 -*-
"""
Build boxed 16-frame context per video (arytenoids & epiglottis) using your CNN detector.

Hard-coded paths (as requested):
  JSON_PATH   = /research/home/he234993/platypus/all_data/all_data_correct_ts.json
  WEIGHT_PATH = /research/home/he234993/last_model.pth
  OUT_ROOT    = /research/home/he234993/platypus/Results

Output:
  OUT_ROOT/<video_stem>/
      arytenoids/ 00_frame_01178.jpg ... (16)
      epiglottis/ 00_frame_01367.jpg ... (16)

Rule:
  indices = start_idx + i*stride (i=0..15, stride=3)
  If idx out of range OR frame read fails -> fallback to start_idx (repeat first frame).
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
from torchvision.ops import nms

from config import CLASSES, DEVICE
from model import create_model


# -------------------- HARD-CODED PATHS --------------------
JSON_PATH = "/research/home/he234993/platypus/all_data/all_data_correct_ts.json"
WEIGHT_PATH = "/research/home/he234993/last_model.pth"
OUT_ROOT = "/research/home/he234993/platypus/Results"

# 如果 JSON 里 video_path 不是绝对路径（像 "xxx.mp4"），会按下面这些候选根目录去拼
VIDEO_ROOT_CANDIDATES = [
    "/research/home/he234993/platypus/all_data/all_video",
]
# ---------------------------------------------------------


# -------------------- SAMPLING CONFIG --------------------
NUM_FRAMES = 16
STRIDE = 3
# ---------------------------------------------------------


# -------------------- DETECTOR CONFIG --------------------
SCORE_THR = 0.15
SCORE_THR_FALLBACK = 0.05
NMS_IOU = 0.5
PAD_RATIO = 0.15
# ---------------------------------------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clear_jpgs(folder: str):
    if not os.path.isdir(folder):
        return
    for f in Path(folder).glob("*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass


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


def resolve_video_path(video_path: str) -> str:
    vp = (video_path or "").strip()
    if not vp:
        return ""

    if os.path.isabs(vp) and os.path.isfile(vp):
        return vp

    # try candidates
    for root in VIDEO_ROOT_CANDIDATES:
        if root:
            cand = os.path.join(root, vp)
        else:
            cand = vp
        if os.path.isfile(cand):
            return cand

    return ""


def read_frame_by_index(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def build_indices_with_fallback(start_idx: int, num_frames: int, stride: int, total_frames: int) -> List[int]:
    """
    indices = start + i*stride
    if idx out of range -> start
    """
    start_idx = int(max(0, start_idx))
    out = []
    for i in range(int(num_frames)):
        idx = start_idx + i * int(stride)
        if total_frames > 0 and idx >= total_frames:
            idx = start_idx
        out.append(int(idx))
    return out


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


def process_one_organ(
    model,
    video_abs_path: str,
    out_dir: str,
    organ_name: str,
    target_id: int,
    target_label: str,
    start_idx: int,
):
    ensure_dir(out_dir)
    clear_jpgs(out_dir)  # avoid mixing old files

    cap = cv2.VideoCapture(video_abs_path)
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {video_abs_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = build_indices_with_fallback(int(start_idx), NUM_FRAMES, STRIDE, total_frames)

    for i, idx in enumerate(indices):
        frame = read_frame_by_index(cap, idx)

        # fallback to first frame if read fails
        if frame is None:
            frame = read_frame_by_index(cap, int(start_idx))

        # still None: use dummy black image, but keep 16 outputs
        if frame is None:
            print(f"[WARN] read fail even fallback -> dummy. organ={organ_name} idx={idx} start={start_idx} video={video_abs_path}")
            frame = np.zeros((512, 512, 3), dtype=np.uint8)

        # detect (two thresholds) + fallback to whole-frame box
        try:
            box, sc = detect_one_box(model, frame, target_id, SCORE_THR, NMS_IOU, PAD_RATIO)
        except Exception:
            try:
                box, sc = detect_one_box(model, frame, target_id, SCORE_THR_FALLBACK, NMS_IOU, PAD_RATIO)
            except Exception:
                H, W = frame.shape[:2]
                box, sc = [0, 0, W - 1, H - 1], 0.0

        vis = draw_box(frame, box, target_label, sc)
        out_name = f"{i:02d}_frame_{idx:05d}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, vis)

    cap.release()
    print(f"[DONE] {organ_name}: {out_dir}")


def get_start_idx_from_sam2_frames(sam2_frames: Dict[str, Any], organ_key: str) -> Optional[int]:
    """
    Prefer best_frame_idx; fallback to best_block_frames[0] if needed.
    """
    info = (sam2_frames or {}).get(organ_key, {}) or {}
    if "best_frame_idx" in info and info["best_frame_idx"] is not None:
        return int(info["best_frame_idx"])
    bbf = info.get("best_block_frames", []) or []
    if len(bbf) > 0 and bbf[0] is not None:
        return int(bbf[0])
    return None


def main():
    print("[INFO] JSON_PATH   =", JSON_PATH)
    print("[INFO] WEIGHT_PATH =", WEIGHT_PATH)
    print("[INFO] OUT_ROOT    =", OUT_ROOT)

    ary_id = find_target_class_id(CLASSES, "arytenoid")
    epi_id = find_epiglottis_class_id(CLASSES)

    if ary_id is None or epi_id is None:
        print("[FATAL] Cannot find class id in CLASSES.")
        print("CLASSES =", list(enumerate(CLASSES)))
        raise SystemExit(1)

    print(f"[INFO] ary_id={ary_id} class={CLASSES[ary_id]}")
    print(f"[INFO] epi_id={epi_id} class={CLASSES[epi_id]}")

    model = load_detector(WEIGHT_PATH)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen_videos = set()
    n_ok = 0

    for sample in data:
        video_path = sample.get("video_path", "") or sample.get("video_full_path", "")
        video_abs = resolve_video_path(video_path)
        if not video_abs:
            # print once per missing
            print(f"[WARN] video not found for video_path={video_path}")
            continue

        # de-duplicate per video
        if video_abs in seen_videos:
            continue
        seen_videos.add(video_abs)

        video_stem = Path(video_abs).stem
        out_video_dir = os.path.join(OUT_ROOT, video_stem)

        sam2_frames = sample.get("sam2_frames", {}) or {}
        ary_start = get_start_idx_from_sam2_frames(sam2_frames, "arytenoids")
        epi_start = get_start_idx_from_sam2_frames(sam2_frames, "epiglottis")

        if ary_start is None and epi_start is None:
            print(f"[WARN] no best_frame_idx for both organs: {video_stem}")
            continue

        # Only create these two folders (no _tmp_frames16 / no masks_raw)
        if ary_start is not None:
            out_ary = os.path.join(out_video_dir, "arytenoids")
            process_one_organ(
                model=model,
                video_abs_path=video_abs,
                out_dir=out_ary,
                organ_name="arytenoids",
                target_id=int(ary_id),
                target_label=CLASSES[ary_id],
                start_idx=int(ary_start),
            )
        else:
            print(f"[WARN] arytenoids missing best_frame_idx: {video_stem}")

        if epi_start is not None:
            out_epi = os.path.join(out_video_dir, "epiglottis")
            process_one_organ(
                model=model,
                video_abs_path=video_abs,
                out_dir=out_epi,
                organ_name="epiglottis",
                target_id=int(epi_id),
                target_label=CLASSES[epi_id],
                start_idx=int(epi_start),
            )
        else:
            print(f"[WARN] epiglottis missing best_frame_idx: {video_stem}")

        n_ok += 1
        print(f"[VIDEO OK] {video_stem}")

    print(f"[ALL DONE] processed videos = {n_ok}, unique_video_paths = {len(seen_videos)}")


if __name__ == "__main__":
    main()
