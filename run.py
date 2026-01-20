import os, glob
import cv2
import torch
import numpy as np
from torchvision.ops import nms

from config import CLASSES, DEVICE, OUT_DIR
from model import create_model

# ===================== 你只需要改这里 =====================
TARGET_NAME = "arytenoid"   # "arytenoid" /"epiglotis"
img_path = "/content/drive/MyDrive/train data/laryngoscope-labeling/train/larynx16_jpg.rf.649e43e018470a10934b98adc0376854.jpg"
SCORE_THR = 0.15             # 召回优先就低一点
NMS_IOU = 0.5
PAD_RATIO = 0.15             # 框扩大比例（只要框到就可以，扩大点更稳）
# =========================================================

# -------- 0) 找到目标类别ID（支持“包含匹配”，避免你CLASSES里名字很怪） --------
def find_target_class_id(classes, target_name: str):
    t = (target_name or "").lower().strip()
    # 先精确匹配
    for i, n in enumerate(classes):
        if (n or "").lower().strip() == t:
            return i
    # 再包含匹配（更适合你现在 'epiglottisrotation' 这种）
    hits = [i for i, n in enumerate(classes) if t in (n or "").lower()]
    return hits[0] if hits else None

target_id = find_target_class_id(CLASSES, TARGET_NAME)
if target_id is None:
    print("[ERR] 在 CLASSES 里找不到目标类：", TARGET_NAME)
    print("CLASSES =", list(enumerate(CLASSES)))
    raise SystemExit

print(f"[INFO] TARGET_NAME={TARGET_NAME} -> target_id={target_id}, class_name={CLASSES[target_id]}")

# -------- 1) 找到你训练出来的权重 --------
weight_path = "/research/home/he234993/PCHCNN/last_model.pth"
assert os.path.isfile(weight_path), f"权重不存在: {weight_path}"
print("[INFO] Use weight:", weight_path)


# -------- 2) 加载模型 --------
num_classes = len(CLASSES)
model = create_model(num_classes=num_classes)  # 某些repo可能是 create_model(num_classes)，你这里按你repo为准
ckpt = torch.load(weight_path, map_location="cpu")
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)
model.to(DEVICE).eval()

# -------- 3) 读图并推理 --------
orig = cv2.imread(img_path)
assert orig is not None, f"读图失败: {img_path}"

rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    out = model(tensor)[0]

boxes  = out["boxes"].detach().cpu()
labels = out["labels"].detach().cpu()
scores = out["scores"].detach().cpu()

# -------- 4) conf过滤 + 类别过滤 --------
keep = scores >= SCORE_THR
boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

# 只保留目标类别
cls_keep = labels == int(target_id)
boxes, labels, scores = boxes[cls_keep], labels[cls_keep], scores[cls_keep]

if len(boxes) == 0:
    print(f"[WARN] No det for class={CLASSES[target_id]} above thr={SCORE_THR}")
    print("[DBG] 你可以把 SCORE_THR 再降一点，比如 0.05")
    raise SystemExit

# -------- 5) NMS + 取最高分 --------
keep_idx = nms(boxes, scores, iou_threshold=NMS_IOU)
boxes, labels, scores = boxes[keep_idx], labels[keep_idx], scores[keep_idx]

top = torch.argmax(scores)
bb = boxes[top].to(torch.int).tolist()
cls_id = int(labels[top])
sc = float(scores[top])

# -------- 6) padding 放大框（可选但推荐）--------
H, W = orig.shape[:2]
x1, y1, x2, y2 = bb
w, h = max(1, x2 - x1), max(1, y2 - y1)

pad = float(PAD_RATIO)
x1 = max(0, int(x1 - pad * w))
y1 = max(0, int(y1 - pad * h))
x2 = min(W - 1, int(x2 + pad * w))
y2 = min(H - 1, int(y2 + pad * h))

# -------- 7) 画框 + 保存 --------
vis = orig.copy()
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
cv2.putText(
    vis,
    f"{CLASSES[cls_id]} {sc:.2f}",
    (x1, max(0, y1 - 6)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2,
    cv2.LINE_AA,
)

out_dir = "/content/infer"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "single_result.jpg")
cv2.imwrite(out_path, vis)

print("[OK] Saved:", out_path)
print("[OK] box:", [x1, y1, x2, y2], "cls:", CLASSES[cls_id], "score:", sc)
