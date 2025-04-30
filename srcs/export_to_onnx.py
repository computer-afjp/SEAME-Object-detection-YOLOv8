from ultralytics import YOLO

model = YOLO("yolov5s.pt")
model.export(format="onnx", imgsz=640, opset=12, dynamic=True)
