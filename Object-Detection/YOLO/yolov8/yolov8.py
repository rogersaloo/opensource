import os 
import sys
from pathlib import Path
import importlib
import wget
from ultralytics import YOLO
from abstract_base_model import BaseModel


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


sys.path.insert(0, os.path.abspath(
os.path.join(os.path.dirname(__file__), './../../')
))

class Model(BaseModel):
    NAME = "yolo8"

    def __init__(self, opt=None):
        super().__init__(opt)
        self.opt = opt
    
    def build_model(self, model_path=None):
        if model_path is not None:
            self.path = model_path
            print("using custom weighhts", self.path)
        else:
            version=self.opt.version

            # Define a dictionary mapping YOLO version letters to their corresponding model names
            model_names_pt = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt', 'l':'yolov8l.pt','x':'yolov8x.pt'}
            #Construct the path to the corresponding  YOLO model file
            self.path = f'model/yolov8{version}.pt'
            self.path = self.opt.weights
            print("Using default model weights:", self.path)

        if os.path.getsize(self.path) > 0:
          path = self.path
          print("Loading model weights from file:", path)
        else:
          print("Downloading model weights")
          URL = "https://ecos-ai-test-upload.s3.ap-northeast-1.amazonaws.com/models/yolo8/pretrained_weight/model_weight.pt"
          path = wget.download(URL)
          print("Downloaded model weights to file:", path)

        self.model=YOLO(self.path)

        return self.model

        def fit(self):
            """
        Args:
          ****minimun value******
          data(str): input data.yaml
          epochs(int): number of epochs
          imgsz(int): Image size (640)
          device:0,1,2,3 for GPU or device:cpu
          batch_size: a number of samples processed before the model is updated
          **************************************************************

        Returns:
            model_path(str): default "runs/detect/train/weights/best.pt"
        """

        return self.model.train(data=self.opt.data, epochs=self.opt.epochs,
                                imgsz=self.opt.imgsz, device=self.opt.device,
                                batch=self.opt.batch_size)
    
    def validate(self):
        """
        Args:
        Returns:
            metrics(list): Validated results
        """
        
        # Validate the model
        metrics =self.model.val()   # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        if self.opt.task == "segment":
            metrics.seg.map    # map50-95(M)
            metrics.seg.map50  # map50(M)
            metrics.seg.map75  # map75(M)
            metrics.seg.maps   # a list contains map50-95(M) of each category
        print("validation succeed")
        return metrics

    def predict(self, source=None):
        """
          Args:
              source(str): file or folder for testing
          Returns:
              folder_path(str): runs/detect/predict
        """
        if source is not None:
          self.source = source
          print("custom Image or image folder ", self.source )
        else:
          self.source  = os.path.join(str(ROOT), self.opt.source)
          print("Using default model weights:", self.source )
        return self.model(self.source,save=self.opt.save)