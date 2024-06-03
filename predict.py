from ultralytics import YOLO

model = YOLO("run/yrcc2-dataloader/weights/best.pt")

for result in model.predict(
        source="datasets/2024new_dataset",
        imgsz=768,
        batch=10,
        device="0",
        save=True,
        stream=True,
        project="predict",
        name="2024new"
):
    result.save()
    pass
