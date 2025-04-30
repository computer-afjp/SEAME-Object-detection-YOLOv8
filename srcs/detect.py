from ultralytics import YOLO


def run_prediction() -> None:
    model = YOLO("yolov8n.pt")
    model.predict(
        "bus.jpg",
        save=True, 
        imgsz=256, 
        conf=0.5
    )


if __name__ == "__main__":
    run_prediction()
