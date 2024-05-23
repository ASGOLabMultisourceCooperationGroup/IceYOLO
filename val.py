from ultralytics import YOLO

model = YOLO("weights/best.pt")

model.val(
    project="run",
    data="datasets/dataset_yrcc2.yaml",
    imgsz=768,
    device="0",
    workers=0,
)