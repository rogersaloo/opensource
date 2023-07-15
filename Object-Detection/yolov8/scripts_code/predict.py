from ultralytics import YOLO

def load_train_model(type):
        
      

        #load
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
        model_path= f'model/yolov8{version}.pt' #to load official model 
        model_weight='runs/detect/train/weights/best.pt'
       

        
        if type=="official":
            #official model 
            return YOLO(model_path)
        
        elif type=="custom":
            #load custom model
            return YOLO(model_weight)
        else:
            return "Invalide case type "
        

def prediction ():
         # Load a model
    type="custom"
    model=load_train_model(type)


    # Predict with the model
    """
    Key	        |Value	    |Description
    data	    |None	    |path to data file, i.e. coco128.yaml
    imgsz	    |640	    |image size as scalar or (h, w) list, i.e. (640, 480)
    batch	    |16	        |number of images per batch (-1 for AutoBatch)
    save_json	|False	    |save results to JSON file
    save_hybrid	|False	    |save hybrid version of labels (labels + additional predictions)
    conf	    |0.001	    |object confidence threshold for detection
    iou	        |0.6	    |intersection over union (IoU) threshold for NMS
    max_det	    |300	    |maximum number of detections per image
    half	    |True	    |use half precision (FP16)
    device	    |None	    |device to run on, i.e. cuda device=0/1/2/3 or device=cpu
    dnn	        |False      |use OpenCV DNN for ONNX inference
    plots	    |False	    |show plots during training
    rect	    |False	    |support rectangular evaluation
    split	    |val	    |dataset split to use for validation, i.e. 'val', 'test' or 'train'

    """
    test_path='https://ultralytics.com/images/bus.jpg' #test folder  path
    results = model(test_path,save=True)  # predict on an image
    print("prediction succeed")

    


            

if __name__=='__main__':
    prediction()
