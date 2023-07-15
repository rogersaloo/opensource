import os 
import sys
from pathlib import Path
import importlib
import wget
from ultralytics import YOLO


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# Load a model

model = YOLO("yolov8n.pt")
results = model("https://ultralytics.com/images/bus.jpg")