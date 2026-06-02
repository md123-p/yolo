from ultralytics import YOLO
import warnings
import argparse
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in scalar divide')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default="/media/ubuntu/DATA/md/YOLOv26n/runs/yolov3-tiny/weights/best.pt", help='loading weights')
    return parser.parse_args()

if __name__ == '__main__':
    args=main()
    model = YOLO(args.weights)
    print("testing TensorRT")
    results=model.val(
              data="antiuav.yaml",
              batch=1, 
              imgsz=640,
              device=0,
              half=True,

              project="/media/ubuntu/DATA/md/YOLOv26n/runs/detect/models",
              name='yolo3')
    
    speed = results.speed
    preprocess = speed['preprocess']
    inference = speed['inference']
    postprocess = speed['postprocess']
    total_ms = preprocess + inference + postprocess
    fps = 1000.0 / total_ms

    print("\n" + "="*40)
    print(f"?? final result (TensorRT Engine)")
    print("="*40)
    print(f"Preprocessing time: {preprocess:.2f} ms")
    print(f"Model inference time: {inference:.2f} ms")
    print(f"Post-processing time: {postprocess:.2f} ms")
    print("-" * 40)
    print(f"Single frame delay: {total_ms:.2f} ms")
    print(f"real FPS:   {fps:.2f}")
    print("="*40 + "\n")
























