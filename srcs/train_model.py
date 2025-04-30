from ultralytics import YOLO

# Load a pretrained (official) YOLO model
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' which is a default trained model.
results = model.train(
    data="coco8.yaml", # We need to set our own at seame_obj_detection.yaml
    epochs=3,
    imgsz=256,
    batch=16,
    name="object_detection_v1"
)

# Evaluate the model's performance on the validation set
results = model.val()
    
# Export the model to ONNX format
success = model.export(format="onnx")