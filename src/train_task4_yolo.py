import os
from ultralytics import YOLO

def main():
    print("Starting Task 4 Training Pipeline==-->")
    
    # 1. Create the YAML config so YOLO knows where the 1,500 images are
    # Using absolute paths to prevent any 'File Not Found' errors
    project_root = os.getcwd()
    yaml_path = os.path.join(project_root, 'data', 'task4_data.yaml')
    
    yaml_content = f"""
path: {project_root}/data/task4_train
train: images/train
val: images/train

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"📝 YAML config created at: {yaml_path}")

    # 2. Load the Pre-trained YOLOv11 Nano model
    print("📥 Loading YOLOv11n model...")
    model = YOLO('yolo11n.pt') 

    # 3. Start training using Apple Silicon (M2) Acceleration
    print("Training starting on MPS (Apple Silicon GPU)===>")
    model.train(
        data=yaml_path, 
        epochs=30, 
        imgsz=640, 
        batch=16, 
        device='mps', # Crucial for M2 speed
        name='yolo_task4_finetune'
    )

if __name__ == "__main__":
    main()