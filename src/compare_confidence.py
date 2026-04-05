import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def get_confidences(model_path, data_yaml):
    model = YOLO(model_path)
    # Use predict instead of val to get individual box confidence scores
    # We sample 300 images to match our Task 2 robustness standard
    results = model.predict(data=data_yaml, device='mps', conf=0.25, imgsz=640, stream=True)
    
    confidences = []
    print(f"Extracting scores from {model_path}...")
    
    for i, result in enumerate(results):
        if result.boxes is not None:
            confidences.extend(result.boxes.conf.cpu().tolist())
        if i >= 299: # Stop at 300 images for consistency
            break
    return confidences

def main():
    # 1. Baseline: Pre-trained YOLOv11 on Blurred Images
    conf_blurred = get_confidences('yolo11n.pt', 'data/blurred_baseline.yaml')
    
    # 2. Improvement: Fine-tuned YOLOv11 (best.pt) on Restored Images
    conf_restored = get_confidences('runs/detect/yolo_task4_finetune/weights/best.pt', 'data/task4_data.yaml')

    # Visual Comparison Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(conf_blurred, bins=50, alpha=0.5, label='Blurred (Baseline)', color='red', density=True)
    plt.hist(conf_restored, bins=50, alpha=0.5, label='Restored (Fine-tuned)', color='blue', density=True)
    
    plt.title('Task 3: Confidence Distribution Analysis')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    os.makedirs('runs', exist_ok=True)
    output_path = "runs/confidence_comparison_hd.png"
    plt.savefig(output_path)
    print(f"Analysis complete. Histogram saved to: {output_path}")

if __name__ == "__main__":
    main()