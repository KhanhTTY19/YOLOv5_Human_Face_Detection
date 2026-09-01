import sys

import cv2
import torch

sys.path.append("/home/khanhtty/YOLOv5_Human_Face_Detection")
from utils.augmentations import letterbox

model = torch.load("yolov5s.pt", map_location="cpu", weights_only=False)["model"].float().eval()
img_path = "/home/khanhtty/dataset/COCO2017/images/val2017/000000001000.jpg"
img0 = cv2.imread(img_path)

img_scale = letterbox(img0, new_shape=(640, 640))[0]
img_numpy = cv2.cvtColor(img_scale, cv2.COLOR_BGR2RGB)
# cv2.imwrite('custom_loss/scaled_image_preview.jpg', img_scale)

img_tensor = torch.from_numpy(img_numpy).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    model.train()
    train_outputs = model(img_tensor)
    model.eval()

raw_predictions = train_outputs[1] if isinstance(train_outputs, tuple) else train_outputs

print("\n=== RAW OUTPUTS ===")
for i, pred in enumerate(raw_predictions):
    print(f"Detection head {i + 1}: {pred.shape}")

# # ========== THÔNG TIN ANCHORS VÀ STRIDE ==========
stride = [8, 16, 32]
anchors = [
    [[10, 13], [16, 30], [33, 23]],  # P3/8
    [[30, 61], [62, 45], [59, 119]],  # P4/16
    [[116, 90], [156, 198], [373, 326]],  # P5/32
]

print("\n=== ANCHOR CONFIGURATION ===")
for i, (s, anc) in enumerate(zip(stride, anchors)):
    grid_h, grid_w = raw_predictions[i].shape[2:4]
    print(f"Layer {i + 1}: stride={s}, grid={grid_h}x{grid_w}, anchors={anc}")


def decode_yolo_output(pred, stride, anchors):
    """Decode raw YOLO output to bounding box pred: [batch, 3, grid_h, grid_w, 85] stride: 8, 16 ỏ 32 anchors: list of
    [w,h] pairs.

    Returns:
        decode: [batch, 3, grid_h, grid_h, grid_w, 85]
    """
    _batch_size, num_anchors, grid_h, grid_w, _num_channels = pred.shape
    device = pred.device

    yv, xv = torch.meshgrid([torch.arange(grid_h), torch.arange(grid_w)], indexing="ij")
    grid = torch.stack((xv, yv), 2).view(1, 1, grid_h, grid_w, 2).float().to(device)

    anchor_grid = torch.tensor(anchors, device=device).float().view(1, num_anchors, 1, 1, 2)

    decoded = pred.clone()

    decoded[..., 0:2] = (torch.sigmoid(pred[..., 0:2]) * 2 - 0.5 + grid) * stride
    decoded[..., 2:4] = (torch.sigmoid(pred[..., 2:4]) * 2) ** 2 * anchor_grid * stride
    decoded[..., 4:] = torch.sigmoid(pred[..., 4:])

    return decoded


print("\n=== DECODED PREDICTIONS ===")
decoded_predictions = []
for i, pred in enumerate(raw_predictions):
    decoded = decode_yolo_output(pred, stride[i], anchors[i])
    decoded_predictions.append(decoded)

    print(f"Layer {i + 1}:")
    print(f"  XY range: [{decoded[..., 0:2].min():.1f}, {decoded[..., 0:2].max():.1f}]")
    print(f"  WH range: [{decoded[..., 2:4].min():.1f}, {decoded[..., 2:4].max():.1f}]")
    print(f"  Objectness: [{decoded[..., 4].min():.4f}, {decoded[..., 4].max():.4f}]")
    print(f"  Max class prob: {decoded[..., 5:].max():.4f}")
