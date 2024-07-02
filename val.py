from ultralytics import YOLO

model = YOLO("weights/yrcc2-cbam.pt")

model.val(
    project="run",
    data="datasets/dataset_yrcc2.yaml",
    imgsz=768,
    batch=5,
    device="0",
    workers=0,
    save_json=True,
    normalize=255,
    input_channel=3,
)
