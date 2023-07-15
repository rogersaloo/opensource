from ultralytics import YOLO


def load_model(type):

    """
    This function does something with the provided variables.
    """

    

    # Define a dictionary mapping YOLO version letters to their corresponding model names
    model_names_pt = {'n': 'yolov8n.pt', 's': 'yolov8s.pt', 'm': 'yolov8m.pt', 'l':'yolov8l.pt','x':'yolov8x.pt'}
    #Define a dictionary mapping YOLO yaml version letters to their corresponding model names
    model_names_yaml={'n': 'yolov8n.yaml', 's': 'yolov8s.yaml', 'm': 'yolov8m.yaml', 'l':'yolov8l.yaml','x':'yolov8x.yaml'}
    #version
    version='n'
    #Construct the path to the corresponding  YOLO model file
    model_path= f'model/yolov8{version}.pt'
    model_yaml=f'yolov8{version}.yaml'
    """
    "new": loads a new, untrained YOLO model based on the specified YAML file
    "transfer": loads a pre-trained YOLO model and continues training it on a new dataset based on the specified YAML file and model file
    "pretrained": loads a pre-trained YOLO model based on the specified model file

    """
    if type=="new":
        return YOLO(model_yaml)
    elif type=="transfer":
        return YOLO(model_yaml).load(model_path)
    elif type=="pretrained":
        #load pretrained model
        return YOLO(model_path)
    else:
        return "Invalide case type "
    


'''
    Args of YOLO function
    Key	            | Default Value	 |   Description
    model           | None	         |   path to model file, i.e. yolov8n.pt, yolov8n.yaml
    data	        | None	         |   path to data file, i.e. coco128.yaml
    epochs	        | 100	         |   number of epochs to train for (mandatory)
    patience        | 50	         |   epochs to wait for no observable improvement for early stopping of training
    batch	        |  16	         |   number of images per batch (-1 for AutoBatch)
    imgsz	        |  640	         |   size of input images as integer or w,h (mandatory)
    save	        |  True	         |   save train checkpoints and predict results
    save_period	    |  -1	         |   Save checkpoint every x epochs (disabled if < 1)
    cache	        | False	         |   True/ram, disk or False. Use cache for data loading
    device	        | None	         |   device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    workers	        | 8	             |   number of worker threads for data loading (per RANK if DDP)
    project	        | None	         |   project name
    name	        | None	         |   experiment name
    exist_ok	    | False	         |   whether to overwrite existing experiment
    pretrained	    | False	         |   whether to use a pretrained model
    optimizer	    | 'SGD'	         |   optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
    verbose	        | False	         |   whether to print verbose output
    seed	        |  0	         |   random seed for reproducibility
    deterministic   | True           |   whether to enable deterministic mode
    single_cls	    | False	         |   train multi-class data as single-class
    image_weights	| False	         |   use weighted image selection for training
    rect	        | False	         |   support rectangular training
    cos_lr	        |False	         |   use cosine learning rate scheduler
    close_mosaic	|10	             |   disable mosaic augmentation for final 10 epochs
    resume	        |False	         |   resume training from last checkpoint
    amp	            |True	         |   Automatic Mixed Precision (AMP) training, choices=[True, False]
    lr0	            |0.01	         |   initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    lrf	            |0.01	         |   final learning rate (lr0 * lrf)
    momentum	    |0.937	         |   SGD momentum/Adam beta1
    weight_decay	|0.0005	         |   optimizer weight decay 5e-4
    warmup_epochs	|3.0	         |   warmup epochs (fractions ok)
    warmup_momentum	|0.8	         |   warmup initial momentum
    warmup_bias_lr	|0.1	         |   warmup initial bias lr
    box	            |7.5	         |   box loss gain
    cls	            |0.5	         |   cls loss gain (scale with pixels)
    dfl	            |1.5	         |   dfl loss gain
    fl_gamma	    |0.0	         |   focal loss gamma (efficientDet default gamma=1.5)
    label_smoothing	|0.0	         |   label smoothing (fraction)
    nbs	            |64	             |   nominal batch size
    overlap_mask	|True	         |   masks should overlap during training (segment train only)
    mask_ratio	    |4	             |   mask downsample ratio (segment train only)
    dropout	        |0.0	         |   use dropout regularization (classify train only)
'''
    
def train():
    type="pretrained" #new #transfer
    data_yaml='coco128.yaml' # path/to/data.yaml

    #load_model
    model=load_model(type)
    #train model
    model.train(data=data_yaml, epochs=2, imgsz=640)
    print(" train succeed")
    

if __name__=='__main__':
    
    train()

