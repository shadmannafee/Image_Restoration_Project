from ultralytics import YOLO
import os

def main():
    # Load the model that survived the crash
    model_path = 'runs/detect/yolo_task4_finetune/weights/best.pt'
    if not os.path.exists(model_path):
        print("err best.pt not found!")
        return

    model = YOLO(model_path)
    
    print("Running Final Evaluation on 1,500 Restored Images...")
    results = model.val(data='data/task4_data.yaml', device='mps')
    
    # Extracting the 4 Golden Metrics for the report
    stats = results.results_dict
    print("\n" + "="*32)
    print("FINAL PROJECT METRICS")
    print("="*30)
    print(f"mAP50:      {stats['metrics/mAP50(B)']: .4f}")
    print(f"mAP50-95:   {stats['metrics/mAP50-95(B)']: .4f}")
    print(f"Precision:  {stats['metrics/precision(B)']: .4f}")
    print(f"Recall:     {stats['metrics/recall(B)']: .4f}")
    print("="*40)

if __name__ == "__main__":
    main()