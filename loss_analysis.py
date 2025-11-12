import torch
import cv2
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# ==== CONFIG ====
weights = 'runs/train/yolov5s-640/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imgsz = 640
dataset_root = Path('/home/khanhtty/dataset/COCO2017')  # dataset root
val_txt = dataset_root / 'val2017.txt'  
save_dir = Path('./runs/train/yolov5s-640/compare_vis')
save_dir.mkdir(parents=True, exist_ok=True)

model = DetectMultiBackend(weights, device=device)
stride = model.stride
names = model.names

with open(val_txt, 'r') as f:
    lines = f.readlines()

img_paths = []
for line in lines[:100]:
    line = line.strip()
    if line:
        # Nếu là relative path, ghép với dataset_root
        img_path = dataset_root / line if not Path(line).is_absolute() else Path(line)
        img_paths.append(img_path)

print(f"Found {len(img_paths)} images to process")

for img_path in img_paths:
    img_path = Path(img_path)
    file_name = img_path.stem
    print(f"Processing {file_name}...")

    # ==== LOAD IMAGE & LABEL ====
    label_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
    
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        print(f"Cannot read {img_path}, skipping.")
        continue

    # ==== PREPROCESS ====
    img = letterbox(img0, imgsz, stride=stride, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # ==== MODEL INFERENCE ====
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # ==== LOAD GROUND TRUTH ====
    h0, w0 = img0.shape[:2]
    labels = np.loadtxt(label_path).reshape(-1, 5) if Path(label_path).exists() else np.zeros((0, 5))
    if len(labels):
        labels[:, 1:] *= np.array([w0, h0, w0, h0])

    # ==== GROUND TRUTH BOXES ====
    gt_boxes = []
    for c, x, y, w, h in labels:
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        gt_boxes.append((int(x1), int(y1), int(x2), int(y2), int(c)))

    # ==== PREDICTIONS ====
    pred_boxes = pred[0] if pred[0] is not None else torch.empty((0, 6))
    if len(pred_boxes):
        pred_boxes[:, :4] = scale_boxes(img.shape[2:], pred_boxes[:, :4], img0.shape).round()

    # ==== VISUALIZATION ====
    img_show = img0.copy()

    # Predicted boxes (đỏ)
    for *xyxy, conf, cls in pred_boxes:
        label = f'{names[int(cls)]} {conf:.2f}'
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img_show, c1, c2, (0, 0, 255), 2)
        cv2.putText(img_show, label, (c1[0], c1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Ground truth boxes (xanh lá)
    for (x1, y1, x2, y2, c) in gt_boxes:
        label = f'{names[int(c)]} (GT)'
        cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_show, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ==== SAVE OUTPUT ====
    save_path = save_dir / f'{file_name}_compare.jpg'
    cv2.imwrite(str(save_path), img_show)
    print(f"Saved to {save_path}")

print("Done!")