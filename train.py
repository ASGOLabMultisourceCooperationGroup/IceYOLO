# from preprocessor.preprocessor import PreProcessor
from ultralytics import YOLO

backbone_model = "yolov8x-seg-multi.yaml"
# backbone_model = "weights/full.pt"
model = YOLO(backbone_model)
# adapter = PreProcessor()

model.train(
    data=["datasets/dataset_yrcc1.yaml", "datasets/dataset_yrcc2.yaml", "datasets/dataset_yrccms.yaml", "datasets/dataset_albert.yaml"],
    epochs=500,
    batch=30,
    imgsz=768,
    device="0,1",
    patience=40,
    save_period=50,
    workers=0,
    project="multi_run",
    name="test",
    mosaic=0,
    optimizer="AdamW",
    lr0=0.000047,
    resume=False,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    normalize=[255, 255, 1067, 255],
    plots=False,
)
