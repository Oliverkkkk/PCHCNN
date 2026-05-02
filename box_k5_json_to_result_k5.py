# -*- coding: utf-8 -*-
"""
Build boxed K5 frame sets from all_data_correct_ts_k5.json and save them to
Result_K5 with a Result_train-like directory structure.

Output structure:
  /research/home/he234993/platypus/all_data/Result_K5/
      S001/
        arytenoids/
          000206.jpg
          ...
          frames_k5_n16.json
        epiglottis/
          000472.jpg
          ...
          frames_k5_n16.json

Unlike the RAG-frames script, this script reads the selected frame indices from
JSON and fetches those frames from the original video.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from config import CLASSES, DEVICE
from model import create_model


# -------------------- HARD-CODED PATHS --------------------
JSON_PATH = "/research/home/he234993/platypus/all_data/all_data_correct_ts_k5.json"
WEIGHT_PATH = "/research/home/he234993/last_model.pth"
OUT_ROOT = "/research/home/he234993/platypus/all_data/Result_K5"

VIDEO_ROOT_CANDIDATES = [
    "/research/home/he234993/platypus/all_data/all_video",
]
# ---------------------------------------------------------


# -------------------- DETECTOR CONFIG --------------------
SCORE_THR = 0.15
SCORE_THR_FALLBACK = 0.05
NMS_IOU = 0.5
PAD_RATIO = 0.15
# ---------------------------------------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clear_jpgs(folder: Path):
    if not folder.is_dir():
        return
    for f in folder.glob("*.jpg"):
        try:
            f.unlink()
        except Exception:
            pass


def find_target_class_id(classes: List[str], target_name: str) -> Optional[int]:
    t = (target_name or "").lower().strip()
    for i, n in enumerate(classes):
        if (n or "").lower().strip() == t:
            return i
    hits = [i for i, n in enumerate(classes) if t in (n or "").lower()]
    return hits[0] if hits else None


def find_epiglottis_class_id(classes: List[str]) -> Optional[int]:
    for key in ["epiglott", "epiglot"]:
        hits = [i for i, n in enumerate(classes) if key in (n or "").lower()]
        if hits:
            return hits[0]
    return None


def load_detector(weight_path: str):
    assert os.path.isfile(weight_path), f"Weight not found: {weight_path}"
    model = create_model(num_classes=len(CLASSES))
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
    for root in VIDEO_ROOT_CANDIDATES:
        cand = os.path.join(root, vp) if root else vp
        if os.path.isfile(cand):
            return cand
    return ""


def read_frame_by_index(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


@torch.no_grad()
def detect_one_box(
    model,
    bgr_img: np.ndarray,
    target_id: int,
    score_thr: float,
    nms_iou: float,
    pad_ratio: float,
) -> Tuple[List[int], float]:
    h, w = bgr_img.shape[:2]

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
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad = float(pad_ratio)
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w - 1, int(x2 + pad * bw))
    y2 = min(h - 1, int(y2 + pad * bh))

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


def get_sample_id(sample: Dict[str, Any]) -> str:
    for key in ["video_id", "id"]:
        value = sample.get(key)
        if value:
            return str(value).split("_")[0]
    return "unknown_sample"


def get_frames_dirname(frame_indices: List[int]) -> str:
    return f"frames_k5_n{len(frame_indices)}"


def write_sidecar_json(
    out_root: Path,
    sample_id: str,
    organ: str,
    frames_dirname: str,
    sample: Dict[str, Any],
    frame_indices: List[int],
    video_abs: str,
):
    payload = {
        "sample_id": sample_id,
        "organ": organ,
        "frames_dirname": frames_dirname,
        "video_id": sample.get("video_id"),
        "id": sample.get("id"),
        "video_path": sample.get("video_path"),
        "video_abs_path": video_abs,
        "question": sample.get("question"),
        "frame_indices": [int(x) for x in frame_indices],
        "sam2_frames": (sample.get("sam2_frames") or {}).get(organ, {}),
    }
    dst = out_root / sample_id / organ / f"{frames_dirname}.json"
    ensure_dir(str(dst.parent))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_one_organ(
    model,
    out_root: Path,
    sample: Dict[str, Any],
    sample_id: str,
    organ: str,
    target_id: int,
    target_label: str,
    video_abs: str,
) -> int:
    info = ((sample.get("sam2_frames") or {}).get(organ) or {})
    frame_indices = info.get("best_block_frames") or []

    if not frame_indices:
        print(f"[WARN] no best_block_frames: sample={sample_id} organ={organ}")
        return 0

    frames_dirname = get_frames_dirname(frame_indices)
    out_dir = out_root / sample_id / organ
    ensure_dir(str(out_dir))
    clear_jpgs(out_dir)
    write_sidecar_json(out_root, sample_id, organ, frames_dirname, sample, frame_indices, video_abs)

    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        print(f"[ERR] Cannot open video: {video_abs}")
        return 0

    saved = 0
    first_idx = int(frame_indices[0])
    for frame_idx in frame_indices:
        frame = read_frame_by_index(cap, int(frame_idx))
        if frame is None:
            frame = read_frame_by_index(cap, first_idx)
        if frame is None:
            print(f"[WARN] read fail even fallback: sample={sample_id} organ={organ} idx={frame_idx}")
            continue

        try:
            box, sc = detect_one_box(model, frame, target_id, SCORE_THR, NMS_IOU, PAD_RATIO)
        except Exception:
            try:
                box, sc = detect_one_box(model, frame, target_id, SCORE_THR_FALLBACK, NMS_IOU, PAD_RATIO)
            except Exception:
                h, w = frame.shape[:2]
                box, sc = [0, 0, w - 1, h - 1], 0.0

        vis = draw_box(frame, box, target_label, sc)
        out_path = out_dir / f"{int(frame_idx):06d}.jpg"
        cv2.imwrite(str(out_path), vis)
        saved += 1

    cap.release()
    print(f"[DONE] {sample_id}/{organ} -> {saved} imgs")
    return saved


def main():
    json_path = Path(JSON_PATH)
    out_root = Path(OUT_ROOT)
    assert json_path.is_file(), f"JSON_PATH not found: {JSON_PATH}"
    ensure_dir(str(out_root))

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

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_imgs = 0
    total_samples = 0

    for sample in data:
        video_path = sample.get("video_path", "") or sample.get("video_full_path", "")
        video_abs = resolve_video_path(video_path)
        if not video_abs:
            print(f"[WARN] video not found for video_path={video_path}")
            continue

        sample_id = get_sample_id(sample)
        total_samples += 1

        total_imgs += process_one_organ(
            model=model,
            out_root=out_root,
            sample=sample,
            sample_id=sample_id,
            organ="arytenoids",
            target_id=int(ary_id),
            target_label=CLASSES[ary_id],
            video_abs=video_abs,
        )
        total_imgs += process_one_organ(
            model=model,
            out_root=out_root,
            sample=sample,
            sample_id=sample_id,
            organ="epiglottis",
            target_id=int(epi_id),
            target_label=CLASSES[epi_id],
            video_abs=video_abs,
        )

    print(f"[ALL DONE] samples={total_samples}, images={total_imgs}")
    print(f"[OUT] {OUT_ROOT}")


if __name__ == "__main__":
    main()
