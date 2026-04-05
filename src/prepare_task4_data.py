import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# Wiener Parameters from Task 2
K_VAL = 0.1
PSF_SIZE = 5

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.fft.fft2(img)
    kernel_f = np.fft.fft2(kernel, s=img.shape)
    kernel_f = np.conj(kernel_f) / (np.abs(kernel_f)**2 + K)
    return np.uint8(np.clip(np.abs(np.fft.ifft2(dummy * kernel_f)), 0, 255))

def main():
    # 1. Load Metadata for Stratification
    with open('data/blur_metadata.json', 'r') as f:
        meta = json.load(f)['transformations']
    
    levels = {'light': [], 'medium': [], 'heavy': []}
    for fname, d in meta.items():
        if 5 <= d['size'] <= 10: levels['light'].append(fname)
        elif 11 <= d['size'] <= 17: levels['medium'].append(fname)
        elif 18 <= d['size'] <= 25: levels['heavy'].append(fname)

    # 2. Select 1,500 images (500 per level)
    selected = levels['light'][:500] + levels['medium'][:500] + levels['heavy'][:500]
    print(f"Preparing 1,500 images (Stratified: 500 per level)")

    # 3. Setup Folders
    train_img_dir = 'data/task4_train/images/train'
    train_lab_dir = 'data/task4_train/labels/train'
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lab_dir, exist_ok=True)

    psf = np.zeros((PSF_SIZE, PSF_SIZE))
    psf[PSF_SIZE//2, :] = 1 

    for fname in tqdm(selected):
        # Image Restoration
        img = cv2.imread(f"data/val2017_blurred_deterministic/{fname}", 0)
        if img is not None:
            restored = wiener_filter(img, psf, K_VAL)
            cv2.imwrite(f"{train_img_dir}/{fname}", restored)

            # Label Copying
            lab_name = fname.replace('.jpg', '.txt')
            src_lab = f"data/labels_yolo/val2017/{lab_name}" 
            if os.path.exists(src_lab):
                shutil.copy(src_lab, f"{train_lab_dir}/{lab_name}")

    print(f"Success! Data is ready for training in data/task4_train/")

if __name__ == "__main__":
    main()