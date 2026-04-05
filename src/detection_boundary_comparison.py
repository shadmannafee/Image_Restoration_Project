from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load the model
model = YOLO('yolo11n.pt') 

img_path_b = 'data/val2017_blurred_deterministic/000000000776.jpg' 
img_path_r = 'data/task4_train/images/train/000000000776.jpg'

# Load images
img_blurred = cv2.imread(img_path_b)
img_restored = cv2.imread(img_path_r)

if img_blurred is None or img_restored is None:
    print("Error: Check your image paths!")
else:
    res_b = model(img_blurred)[0].plot()
    res_r = model(img_restored)[0].plot()


    h, w = 640, 850 
    res_b_resized = cv2.resize(res_b, (w, h))
    res_r_resized = cv2.resize(res_r, (w, h))


    cv2.putText(res_b_resized, "Blurred Baseline", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(res_r_resized, "Restored (Fine-tuned)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Stack horizontally
    comp = np.hstack((res_b_resized, res_r_resized))

    # Save to runs
    os.makedirs('runs', exist_ok=True)
    cv2.imwrite('runs/failure_case/detection_boundary_comparison2.png', comp)
    print("Success! Boundary comparison saved to runs/detection_boundary_comparison.png")