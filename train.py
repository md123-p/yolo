from ultralytics import RTDETR
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in scalar divide')


if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/26/yolo26-6-DPC3k2-ConverseSample-C2PSCAA.yaml")
    #try:
        #model.load("yolo26n.pt")
        #print(" yolo26n.pt ")
    #except Exception as e:
        #print(f" {e}")

    
    #model = YOLO("yolo26l.pt") 
    #print("Successfully loaded original yolo26l.pt!")
    model.train(
        data="antiuav.yaml",
        
        epochs=300,      
        patience=100,
        batch=32,               
        imgsz=640,
        optimizer='AdamW',     
        lr0=0.001,             
        weight_decay=0.0005,   
   
        mosaic=0.5,
        close_mosaic=10, 
        mixup=0.1,           
        scale=0.5,              
        translate=0.1,          
        device=1,                
        workers=8,
        project="/media/ubuntu/DATA/md/YOLOv26n/runs/",
        name='1',
        plots=True,
        
        pretrained=True           
    )
    #model.train(data="/media/ubuntu/DATA/md/yolov12-main/datasets/DUT Anti-UAV/antiuav.yaml",
                #epochs=300,
                #batch=16,
                #device=1,
                #pretrained=False)