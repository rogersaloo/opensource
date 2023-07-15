import torch
import torchvision
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self).
        -- <get_transform>:                    Get image data transform function for preprocessing
        -- <build_model>:                       build model for detection
    """

    def __init__(self, opt=None):
        """Initialize the BaseModel class.
        """
        self.opt = opt
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU

    @abstractmethod
    def get_transform(self) -> torchvision.transforms.transforms:
        """Get image data transform function for preprocessing

        Returns:
            transform (func): transform function.
                    This function will take input as image path and output 
                    - raw_image: image numpy mat
                    - image_transform: image tensor (pytorch)
                    Example: torchvision.transforms.transforms: data transforms function
        """
        pass

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """build model for detection

        Returns:
            torch.nn.Module: Inherited from model base class
        """
        pass

    @abstractmethod
    def process_predictions(self, net_output, raw_image_path, raw_image_mat, image_transform, save_image_path):
        """post process the output of the net

        Args:
            net_output (_type_): output of detection net
            raw_image_path (str): raw image path
            raw_image_mat (numpy.Array) raw image numpy mat type
            image_transform (func) image_transform function
            save_image_path (str) save image path
            
        Returns:
            save_image_path
                inform that the annotation image has been written successfully
                in the same directory contain the annotation image, the annotation text file will be named "annotated_image.txt"
                each line format (yolo): 
                    class, x, y, w, h, confidence, class_name
        """
        pass