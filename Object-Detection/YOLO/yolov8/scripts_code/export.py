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
        
def export():
     # Load a model
    type="custom"
    model=load_train_model(type)


    # Export the model

    """
    Format	       |format Argument	 |Model	                    |Metadata
    PyTorch	       |   -	         |yolov8n.pt                |	✅
    TorchScript	   |torchscript	     |yolov8n.torchscript	    |   ✅
    ONNX	       |onnx	         |yolov8n.onnx	            |   ✅
    OpenVINO	   |openvino	     |yolov8n_openvino_model/	|   ✅
    TensorRT	   |engine	         |yolov8n.engine	        |   ✅
    CoreML	       |coreml           |yolov8n.mlmodel	        |   ✅
    TF SavedModel  |saved_model	     |yolov8n_saved_model/	    |   ✅
    TF GraphDef	   |pb	             |yolov8n.pb	            |   ❌
    TF Lite	       |tflite	         |yolov8n.tflite	        |   ✅
    TF Edge TPU	   |edgetpu	         |yolov8n_edgetpu.tflite	|   ✅
    TF.js	       |tfjs	         |yolov8n_web_model/	    |   ✅
    PaddlePaddle   |paddle	         |yolov8n_paddle_model/	    |   ✅
        
    """
    model.export(format='onnx')
    print("successful")
     
if __name__=='__main__':
     
     export()

    