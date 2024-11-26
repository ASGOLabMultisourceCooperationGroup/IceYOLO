import os
from ultralytics import YOLO

backbone_model = "yolov8x-seg-backboneattn.yaml"
#backbone_model = "run/conv-cbam-yrcc2-0/weights/epoch400.pt"

model = YOLO(backbone_model)

if os.name == 'nt':
    model.train(
        data="datasets/dataset_yrcc2.yaml",
        epochs=500,
        batch=1,
        imgsz=768,
        device="0",
        patience=40,
        save_period=50,
        workers=0,
        project="run",
        name="backboneattn",
        mosaic=0
    )
else:
    model.train(
    #    resume=True,
        data="datasets/dataset_yrcc2.yaml",
        epochs=500,
        batch=30,
        imgsz=768,
        device="0,1",
        patience=400,
        save_period=50,
        # workers=0,
        project="run",
        name="backboneattn-yrcc2",
        mosaic=0,
        # normalize=1067,
        # hsv_h=0,
        # hsv_s=0,
        # hsv_v=0
    )