# Experimental Expansion: Scaling the Baseline (Task 2 & 3)

**1. Justification of Increase of Sample Size.**

To first test the Wiener Filter implementation, a 20 image pilot study was developed. Although there was an evident trend in the results, the sample was not large enough to explain the high variation of COCO dataset categories (e.g., small objects vs. large landscapes).

_Action:_ The test set was expanded from 20 to 100 unique images (val2017 IDs 00139 to 09891).

_Objective_:\* To stabilize the Mean PSNR/SSIM and make sure that the decline in the performance of the object detector (YOLOv11) was a systemic issue of classical restoration, rather than a small sample anomaly.

**2. Quantitative Results (N=100)**

Metric Blurred (Input) Restored(Wiener) Delta (Δ)

Mean PSNR 22.97 dB 19.90 dB -3.07 dB
Mean SSIM 0.6461 0.5670 -0.0791
Max SSIM 0.9220 0.8991 -0.0229
Min SSIM. 0.3312 0.1959 -0.1353

**3. Observations & Reaction Analysis**

A. The "Mathematical Penalty"
The increase in the sample size validated a steady reduction in SSIM (12). The Wiener filter is a global operation and the fact that the restoration was pixel-wise made it react negatively to the metrics of the restoration. The filter was over-corrected in high-entropy regions (busy backgrounds) resulting in a noise pattern of a checkerboard that is severely punished by the SSIM index.

B. Detection Sensitivity (The AI Reaction)
Inferential Degradation was a noticeable reaction when the response to 100 images was processed by YOLOv11.
Edge Confusion: The ringing artifacts overshadowed the sharper edges developed by the filter.
Confidence Loss: The confidence scores of the detector decreased due to the manipulation of the artifacts that distorted the feature maps that the neural network relies on to detect objects. As an example, the view of a person with lines around their silhouette ringing is no longer the type of data that YOLO was trained on, as it was considered to be clean.

C. Min vs Max Outlier Analysis.
When the number of images was increased to 100, it turned out that the Wiener filter works best with low-depth and high-contrast pictures (Max SSIM 0.89). On the other hand, it responds poorly to low-light or high-occlusion images (Min SSIM 0.19) in which the motion blur is non-linear and not reflecting our 5x5 non-deterministic assumption.

4. The 100 image expansion was a success and confirmed the baseline. We have statistically confirmed that classical Wiener restoration is counter-productive to automated downstream tasks such as object detection. This gives us a sound justification to move to Task 5 where we will apply a Deep Learning-based method to image restoration without these counter-productive artifacts.
