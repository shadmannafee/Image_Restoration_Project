import cv2
import os
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def main():
    original_dir = "data/val2017"
    restored_dir = "data/task4_train/images/train"
    metadata_path = "data/blur_metadata.json"
    
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('transformations', {})
    
    if not metadata:
        print("Error: 'transformations' key not found in metadata.")
        return

    # Light: size <= 10 | Medium: 10 < size <= 20 | Heavy: size > 20
    tiers = {'light': [], 'medium': [], 'heavy': []}
    for img_name, info in metadata.items():
        size = info.get('size', 0)
        if size <= 10:
            level = 'light'
        elif size <= 20:
            level = 'medium'
        else:
            level = 'heavy'
        tiers[level].append(img_name)

    overall_psnr = []
    overall_ssim = []

    print("Starting Stratified Quantitative Evaluation (300 Images total)...")

    for level in ['light', 'medium', 'heavy']:
        sample_files = tiers[level][:100]
        if not sample_files:
            print(f"Warning: No images found for {level} tier.")
            continue
            
        level_psnr = []
        level_ssim = []

        for filename in sample_files:
            orig = cv2.imread(os.path.join(original_dir, filename), 0)
            rest = cv2.imread(os.path.join(restored_dir, filename), 0)
            
            if orig is not None and rest is not None:
                if orig.shape != rest.shape:
                    orig = cv2.resize(orig, (rest.shape[1], rest.shape[0]))
                
                level_psnr.append(psnr(orig, rest))
                level_ssim.append(ssim(orig, rest))
        
        if level_psnr:
            print(f"Results for {level.upper()} intensity:")
            print(f"  Avg PSNR: {np.mean(level_psnr):.2f} dB")
            print(f"  Avg SSIM: {np.mean(level_ssim):.4f}")
            overall_psnr.extend(level_psnr)
            overall_ssim.extend(level_ssim)

    print("\n" + "="*40)
    print("FINAL TASK 2 QUANTITATIVE SUMMARY")
    print("="*40)
    if overall_psnr:
        print(f"Total Average PSNR: {np.mean(overall_psnr):.2f} dB")
        print(f"Total Average SSIM: {np.mean(overall_ssim):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()