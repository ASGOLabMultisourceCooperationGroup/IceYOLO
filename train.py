from ultralytics import YOLO

model = YOLO("yolov8x-seg-original.yaml")

model.train(
    data="datasets/dataset_yrcc2.yaml",
    epochs=500,
    batch=2,
    imgsz=768,
    device="cuda",
    save_period=50,
    workers=0,
    project="run"
)