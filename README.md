# Image Restoration & Object Detection (Course: COMP 6001)

## Project Overview

This project evaluates the impact of motion blur on object detection accuracy. We develop a pipeline to restore images using both classical (Wiener Filter) and Deep Learning (YOLO/Restormer) approaches, measuring performance improvements via mAP (mean Average Precision).

## Environment & Setup

### **Dataset Acquisition**

To ensure data integrity, the **COCO 2017** dataset was acquired via the terminal using the `wget -c` command. This approach was selected to handle the ~1.2GB file size reliably by supporting interrupted downloads.

- **Sharp Images:** COCO 2017 Val
- **Blurred Images:** COCOBlur (Kaggle)

### **Local Installation**

1. Clone the repo: `git clone [URL]`
2. Install dependencies: `pip install -r requirements.txt`
3. The /data directory is not tracked by Git. Please download the COCO datasets manually to this folder before running scripts.

## Project Structure

- `/src`: Modular Python scripts for deblurring and detection logic.
- `/ai_logs`: Documentation of AI-assisted development sessions.
- `/notebooks`: Experimental results and data visualization.

## AI Attribution

This project utilizes **Gemini** as a technical collaborator. All AI-generated code is reviewed for Mac Silicon compatibility and documented in the `/ai_logs` directory.
