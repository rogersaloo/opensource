from ultralytics import YOLO

# Load a model

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")