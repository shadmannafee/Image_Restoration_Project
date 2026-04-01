import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def wiener_filter(img, kernel, K=0.1):
    """
    Applies a Wiener filter to an image to reverse motion blur.
    K is the noise-to-signal ratio; higher K reduces noise but adds blur.
    """
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel)**2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return np.uint8(np.clip(dummy, 0, 255))

def process_and_compare(image_id, show_plot=False):
    blurred_path = f'data/val2017_blurred_deterministic/{image_id}.jpg'
    sharp_path = f'data/val2017/{image_id}.jpg'
    output_dir = 'data/restored_classical'
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(blurred_path) or not os.path.exists(sharp_path):
        return None

    # Load images in grayscale for classical frequency analysis
    img_blurred = cv2.imread(blurred_path, 0)
    img_sharp = cv2.imread(sharp_path, 0)

    # Point Spread Function (PSF)
    # This represents a 15-pixel horizontal motion blur
    psf = np.zeros((5, 5))
    psf[2, :] = 1.0 
    psf /= psf.sum()

    # Apply the Restoration
    img_restored = wiener_filter(img_blurred, psf, K=0.1)

    # Calculate Metrics 
    score_blurred = psnr(img_sharp, img_blurred)
    score_restored = psnr(img_sharp, img_restored)
    ssim_blurred = ssim(img_sharp, img_blurred, data_range=255)
    ssim_restored = ssim(img_sharp, img_restored, data_range=255)


    # Reult can be used for task 4
    cv2.imwrite(f'{output_dir}/{image_id}_restored.jpg', img_restored)

    # Visual Comparison
    if show_plot:
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(img_blurred, cmap='gray'), plt.title(f'Blurred ({score_blurred:.2f}dB)')
        plt.subplot(132), plt.imshow(img_restored, cmap='gray'), plt.title(f'Restored (SSIM: {ssim_r:.4f})')
        plt.subplot(133), plt.imshow(img_sharp, cmap='gray'), plt.title('Sharp')
        plt.tight_layout()
        plt.show()
    return score_blurred, score_restored, ssim_blurred, ssim_restored

if __name__ == "__main__":
    blurred_folder = 'data/val2017_blurred_deterministic'
    all_images = sorted([f.replace('.jpg', '') for f in os.listdir(blurred_folder) if f.endswith('.jpg')])[:20]
    
    results = []
    print(f"🚀 Processing {len(all_images)} images for Task 2...")

    for i, img_id in enumerate(all_images): #first image should be seen
        metrics = process_and_compare(img_id, show_plot=(i == 0))
        if metrics:
            results.append({
                "Image_ID": img_id,
                "PSNR_Blurred": metrics[0],
                "PSNR_Restored": metrics[1],
                "SSIM_Blurred": metrics[2],
                "SSIM_Restored": metrics[3]
            })
            print(f"✅ [{i+1}/20] {img_id} processed.")
    df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/classical_baseline_results.csv', index=False)
    
    print("\n--- BATCH COMPLETE ---")
    print(df.describe().loc[['mean', 'min', 'max']])