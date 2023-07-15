from yaml_file import  creation_yaml
from train import train
from val  import validation

from predict import prediction
from export import export




if __name__=='__main__':
    """
    #TODO: Folder structure in oder to create our own yaml file in yaml_file.py 
    model_dir=os.path.abspath(f'../yolov8/data/')
    images_dir=os.path.abspath(f"../yolov8/data/images")
    labels_dir=os.path.abspath(f"../yolov8/data/labels")
    execute the yaml_file for creation 
    python yaml_file.py
    """
    
    #if you want to use our own data change the yaml file in the train function 
    
    train()

    validation()
    prediction()
    export()