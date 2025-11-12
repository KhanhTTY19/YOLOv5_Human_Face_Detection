# Recalculate confusion matrix
from pathlib import Path

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# ------------------------ #
#  Utility Functions
# ------------------------ #


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def bbox_iou(box1, box2, eps=1e-7):
    """Compute IoU between box1 (N,4) and box2 (M,4) Returns: (N, M) matrix.
    """
    # box1: (N, 4) -> (N, 1, 4)
    # box2: (M, 4) -> (1, M, 4)
    box1 = box1.unsqueeze(1)  # (N, 1, 4)
    box2 = box2.unsqueeze(0)  # (1, M, 4)

    # Tính intersection
    inter_min = torch.max(box1[..., :2], box2[..., :2])  # (N, M, 2)
    inter_max = torch.min(box1[..., 2:], box2[..., 2:])  # (N, M, 2)
    inter_wh = (inter_max - inter_min).clamp(0)  # (N, M, 2)
    inter = inter_wh[..., 0] * inter_wh[..., 1]  # (N, M)

    # Tính union
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])  # (N, 1)
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])  # (1, M)
    union = area1 + area2 - inter + eps  # (N, M)

    return inter / union  # (N, M)


# ------------------------ #
#  Confusion Matrix Class
# ------------------------ #


class ConfusionMatrix:
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        if detections is None or len(detections) == 0:
            for gc in labels[:, 0].int():
                self.matrix[self.nc, gc] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        if len(detections) == 0:
            for gc in labels[:, 0].int():
                self.matrix[self.nc, gc] += 1
            return

        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = bbox_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if len(x[0]):
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1  # false negative

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # false positive


# ------------------------ #
#  Load model & data
# ------------------------ #

weights = "runs/train/yolov5s-640/weights/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgsz = 640
dataset_root = Path("/home/khanhtty/dataset/COCO2017")
val_txt = dataset_root / "val2017.txt"

model = DetectMultiBackend(weights, device=device)
stride = model.stride
names = model.names
nc = len(names)

cm = ConfusionMatrix(nc, conf=0.25, iou_thres=0.45)

with open(val_txt) as f:
    lines = f.readlines()  # lấy 100 ảnh đầu

for line in lines:
    line = line.strip()
    if not line:
        continue

    img_path = dataset_root / line if not Path(line).is_absolute() else Path(line)
    label_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt"))

    img0 = cv2.imread(str(img_path))
    if img0 is None:
        continue

    img = letterbox(img0, imgsz, stride=stride, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6)

    pred_boxes = pred[0]
    if pred_boxes is not None and len(pred_boxes):
        pred_boxes[:, :4] = scale_boxes(img.shape[2:], pred_boxes[:, :4], img0.shape).round()

    # Ground truth
    if label_path.exists():
        labels = np.loadtxt(label_path).reshape(-1, 5)
    else:
        labels = np.zeros((0, 5))

    if len(labels):
        h0, w0 = img0.shape[:2]
        labels[:, 1:] *= np.array([w0, h0, w0, h0])
        # convert (cx, cy, w, h) → (x1, y1, x2, y2)
        labels_xyxy = np.zeros_like(labels)
        labels_xyxy[:, 0] = labels[:, 0]
        labels_xyxy[:, 1] = labels[:, 1] - labels[:, 3] / 2
        labels_xyxy[:, 2] = labels[:, 2] - labels[:, 4] / 2
        labels_xyxy[:, 3] = labels[:, 1] + labels[:, 3] / 2
        labels_xyxy[:, 4] = labels[:, 2] + labels[:, 4] / 2
        labels = torch.from_numpy(labels_xyxy).to(device)
    else:
        labels = torch.zeros((0, 5), device=device)

    # ----> TÍNH CONFUSION MATRIX
    cm.process_batch(pred_boxes, labels)

print("\nConfusion Matrix:")
print(cm.matrix)
