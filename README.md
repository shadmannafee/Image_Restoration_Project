# Image Restoration & Object Detection (Course: COMP 6001)

## Project Overview

This project evaluates the impact of motion blur on object detection accuracy. We develop a pipeline to restore images using both classical (Wiener Filter) and Deep Learning (YOLO/Restormer) approaches, measuring performance improvements via mAP (mean Average Precision).

## Environment & Setup

### **Dataset Acquisition**

To ensure data integrity, the **COCO 2017** dataset was acquired via the terminal using the `wget -c` command. This approach was selected to handle the ~1.2GB file size reliably by supporting interrupted downloads.

- **Sharp Images:** COCO 2017 Val
- **Blurred Images:** COCOBlur (Kaggle)

### **Local Installation**

1. Clone the repo: `https://github.com/shadmannafee/Image_Restoration_Project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. The /data directory is not tracked by Git. Please download the COCO datasets manually to this folder before running scripts.

## Project Structure

- `/src`: Modular Python scripts for deblurring and detection logic.
- `/ai_logs`: Documentation of AI-assisted development sessions.
- `/notebooks`: Experimental results and data visualization.

## AI Attribution

This project utilizes **Gemini** as a technical collaborator. All AI-generated code is reviewed for Mac Silicon compatibility and documented in the `/ai_logs` directory.

# Third-Party Licenses and Attribution

This project utilizes several open-source components that are subject to their own respective licenses:

**1. Ultralytics YOLOv11:**

- License: AGPL-3.0 License.

- Usage: Used as the primary object detection architecture for baseline evaluation (Task 3) and fine-tuning on restored imagery (Task 4).

**2. Microsoft COCO Dataset (2017):**

- License: Creative Commons Attribution 4.0 (CC BY 4.0).

- Usage: Source images and object annotations were derived from the COCO 2017 validation set to create the blurred and restored experimental domains.

**3. Python Libraries:**

- OpenCV (BSD License): Used for image manipulation and motion blur synthesis.

- Scikit-Image (Modified BSD): Used for calculating PSNR and SSIM metrics.

- Matplotlib (PSF License): Used for generating confidence distribution histograms.

## Hardware Specifications

- **Device:** MacBook Air M2 (16GB Unified Memory)
- **Acceleration:** Apple Metal Performance Shaders (MPS) via `torch.device("mps")`
- **Constraint Management:** Dataset stratified to 1,500 images to ensure the 90-minute training limit was met without hardware thermal throttling.

## Execution Pipeline to reproduce results

### Phase 1: Image Restoration & Metric Analysis (Task 2)

1. **Restoration & Dataset Generation:** `python3 src/deblur_images.py`
   - Applies linear motion blur to the COCO 2017 validation set using parameters from `data/blur_metadata json`.
   - Implements **Wiener Filter** restoration logic to generate the deblurred training and validation samples.
   - **Outputs:** `data/task4_train/images/train`.
2. **Quantitative Metric Evaluation:** `python3 src/compute_restoration_metrics.py`
   - Performs a pixel-wise and structural comparison of restored images against the sharp originals.
   - **Outputs:** A stratified table of **PSNR (dB)** and **SSIM** scores for Light, Medium, and Heavy blur tiers.

### Phase 2: Object Detection Baselines (Task 3)

3. **Blurred Baseline Inference:** `python3 src/baseline_eval.py`
   - Runs the pre-trained YOLO11n model on the synthesized blurred images.
   - Establishes the performance "floor" (mAP 0.1610) required for the comparative study.
4. **Confidence Distribution Mapping:** `python3 src/compare_confidence.py`
   - Extracts detection confidence tensors using the **Apple Metal Performance Shaders (MPS)** backend.
   - **Outputs:** `runs/confidence_comparison_hd.png` to visualize the statistical shift in model certainty post-restoration.

### Phase 3: Dataset Engineering & Fine-tuning (Task 4)

5. **Data/Label Preparation:** `python3 src/prepare_task4_data.py`
   - Converts COCO JSON annotations into YOLO-normalized `.txt` labels.
   - Executes stratified sampling (500 images per tier) and generates the `data/task4_data.yaml` configuration.
6. **M2 Optimized Training:** `python3 src/train_yolo.py`
   - Executes the fine-tuning process on the restored imagery specifically optimized for the MacBook Air M2 hardware.
   - **Saves:** Final weights to `runs/detect/yolo_task4_finetune/weights/best.pt`.

### Phase 4: Comparative Analysis (Task 5)

7. **Final Domain Comparison:** `python3 src/final_eval.py`
   - Validates the fine-tuned model to confirm the final mAP of 0.3604.
   - **Generates:** Precision-Recall (PR) curves and F1-score plots for the critical reflection section of the report.
