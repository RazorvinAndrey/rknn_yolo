from ultralytics import YOLO

SIZE = 640
model = YOLO('yolov8n.pt')
model.train(data='dt_1/data.yaml', epochs=100, device=0, batch=2, imgsz=SIZE, augment=False)
