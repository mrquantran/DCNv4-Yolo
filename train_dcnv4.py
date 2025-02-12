from ultralytics import YOLO

# Build new model from modified config
model = YOLO("/kaggle/working/DCNv4-Yolo/ultralytics/cfg/models/v8/yolov8-dcnv4.yaml")

# Transfer learning (recommended)
# model.load("yolov8n.pt")  # Initialize with pretrained weights

# Train with custom backbone
results = model.train(
    data="/kaggle/working/DCNv4-Yolo/fisheye8k.yaml",
    epochs=100,
    imgsz=640,
    batch=1,
    optimizer="AdamW",
    lr0=0.001,
    task="detection",
)
