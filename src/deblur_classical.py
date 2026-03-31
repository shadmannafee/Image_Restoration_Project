import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio as psnr

def wiener_filter(img, kernel, K=0.01):
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

def process_and_compare(image_id):
    blurred_path = f'data/val2017_blurred_deterministic/{image_id}.jpg'
    sharp_path = f'data/val2017/{image_id}.jpg'
    output_dir = 'data/restored_classical'
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(blurred_path) or not os.path.exists(sharp_path):
        print(f"⚠️ Skipping {image_id}: Files not found.")
        return

    # Load images in grayscale for classical frequency analysis
    img_blurred = cv2.imread(blurred_path, 0)
    img_sharp = cv2.imread(sharp_path, 0)

    # Point Spread Function (PSF)
    # This represents a 15-pixel horizontal motion blur
    psf = np.zeros((15, 15))
    psf[7, :] = 1.0 
    psf /= psf.sum()

    # Apply the Restoration
    img_restored = wiener_filter(img_blurred, psf, K=0.01)

    # Calculate Metrics 
    score_blurred = psnr(img_sharp, img_blurred)
    score_restored = psnr(img_sharp, img_restored)

    print(f"\nResults for {image_id}:")
    print(f"  - PSNR (Blurred): {score_blurred:.2f} dB")
    print(f"  - PSNR (Restored): {score_restored:.2f} dB")

    # Reult can be used for task 4
    cv2.imwrite(f'{output_dir}/{image_id}_restored.jpg', img_restored)

    # Visual Comparison
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(img_blurred, cmap='gray'), plt.title(f'Blurred ({score_blurred:.2f}dB)')
    plt.subplot(132), plt.imshow(img_restored, cmap='gray'), plt.title(f'Wiener Restored ({score_restored:.2f}dB)')
    plt.subplot(133), plt.imshow(img_sharp, cmap='gray'), plt.title('Ground Truth (Sharp)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with a specific image from your val2017 set
    test_image_id = "000000000139" 
    process_and_compare(test_image_id)