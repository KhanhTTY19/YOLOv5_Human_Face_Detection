models=(
  "yolov5m-640"
  "yolov5s-640"
  "yolov5s-640-mixup2"
  "yolov5s-640-no-augment"
  "yolov5s-1024"
)

for model in "${models[@]}"; do
  echo ">>> Running validation for $model ..."
  python val.py \
    --weights "/home/khanhtty/YOLOv5_Human_Face_Detection/runs/train/$model/weights/best.pt" \
    --data person_face_sat.yaml \
    --img 640 --device 0 \
    --name "person-face-sat-coco/conf-thres-41/${model}-confthes61" \
    --conf-thres 0.61 \
    > "runs/val/logs/person-face-sat-coco/${model}-confthes41.log" 2>&1
done
