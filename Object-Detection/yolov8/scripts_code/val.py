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
        
def validation():
     # Load a model
    type="custom"
    model=load_train_model(type)

    # Validate the model
    metrics =model.val()   # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
    print("validation succeed")


     

if __name__=='__main__':
     validation()
   
    
    

     
    