from preprocessor.preprocessor import PreProcessor
from ultralytics import YOLO

backbone_model = "yolov8x-seg-full.yaml"

model = YOLO(backbone_model)
# adapter = PreProcessor()

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
    name="full",
    mosaic=0
)
