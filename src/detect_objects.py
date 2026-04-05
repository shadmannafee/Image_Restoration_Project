import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

def run_detection(model, image_path):
    results = model(image_path, verbose=False)
    result = results[0]
    num_objects = len(result.boxes)
    avg_conf = result.boxes.conf.mean().item() if num_objects > 0 else 0.0
    annotated_img = result.plot()
    return num_objects, avg_conf, annotated_img

def main():
    print("Initializing YOLOv11---->")
    model = YOLO('yolo11n.pt') 
    
    output_dir = 'reports/detections'
    os.makedirs(output_dir, exist_ok=True)
    
    blurred_dir = 'data/val2017_blurred_deterministic'
    restored_dir = 'data/restored_classical'
    
    # Improved ID extraction logic
    files = [f for f in os.listdir(restored_dir) if f.endswith('_wiener_K01_P5.jpg')]
    image_ids = sorted([f.split('_')[0] for f in files])
    
    print(f"Found {len(image_ids)} images to analyze.")
    
    detection_results = []

    for i, img_id in enumerate(image_ids):
        b_path = os.path.join(blurred_dir, f"{img_id}.jpg")
        r_path = os.path.join(restored_dir, f"{img_id}_wiener_K01_P5.jpg")

        # Double check the files exist before running YOLO
        if not os.path.exists(b_path) or not os.path.exists(r_path):
            print(f"Skipping {img_id}: File not found.")
            continue

        b_count, b_conf, b_img = run_detection(model, b_path)
        r_count, r_conf, r_img = run_detection(model, r_path)

        detection_results.append({
            "Image_ID": img_id,
            "Blurred_Count": b_count,
            "Blurred_Conf": round(b_conf, 4),
            "Restored_Count": r_count,
            "Restored_Conf": round(r_conf, 4)
        })

        if i == 0:
            print(f"Generating visual comparison for {img_id}...")
            plt.figure(figsize=(12, 6))
            plt.subplot(121), plt.imshow(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Blurred: {b_count} objects")
            plt.subplot(122), plt.imshow(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Restored: {r_count} objects")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/comparison_sample.png")
            plt.show()

        print(f"[{i+1}/{len(image_ids)}] {img_id} processed.")

    if detection_results:
        df = pd.DataFrame(detection_results)
        df.to_csv('reports/detection_comparison_results.csv', index=False)
        print("\n====TASK 3 ANALYSIS COMPLETE ===")
        print(df.describe().loc[['mean']])
    else:
        print("Error: No detection results were generated.")

if __name__ == "__main__":
    main()