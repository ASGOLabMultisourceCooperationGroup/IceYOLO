from ultralytics import YOLO

model = YOLO("yolov8x-seg-original.yaml")

model.train(
    data="datasets/dataset_yrcc2.yaml",
    epochs=500,
    batch=30,
    imgsz=768,
    device="0,1",
    patience=40,
    save_period=50,
    # workers=0,
    project="run",
    mosaic=0
)
