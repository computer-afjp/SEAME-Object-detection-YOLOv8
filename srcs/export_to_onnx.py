from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=256, opset=11, dynamic=True)
