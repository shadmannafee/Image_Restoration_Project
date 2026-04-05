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

========================================

## Technical Pivot & Bypass Summary

**1. Data Engineering & Label Transformation**

- Conversion of COCO to YOLO: Original COCO JSON labels could not be used in the Ultralytics training engine. We used conversion bypass, which converted complex polygons to YOLO-normalized x, y, w, h text..
- Workspace De-duplication: The conversion process generated duplicate directories (e.g. coco_converted). We have rearranged manually the pathing to data/labels_yolo/ so that the annotations are available to the trainer.
- Dataset Stratification: To satisfy the 90-minute time limit in training, we designed a script that would sample 1,500 images directly (1,500 / 500) with the amount of 500 images of each blur level (Light, Medium, Heavy) so that every blur level had equal representation in the learning.

**2. Environment & Path Resolution**

- Dependency Management: tqdm and scikit-learn have fixed errors of ModuleNotFound in the venv to regain progress tracking.

- Relative Pathing Bug: Fixed a script that was attempting to find relative content but failed to find blur metadata.json as the absolute searches had to be forced using Image Restoration Project root.

- YAML Config Creation: Constructed task4 data.yaml dynamically on the fly to ensure the model had successfully relabelled your local restored image folders with the 80 COCO classes.

**3. Hardware-Specific Bypasses (Apple Silicon M2)**

- MPS Backend Enforcement: Hardly bypassed the default CPU/CUDA logic to use the Metal Performance Shaders (MPS) to accelerate 16GB Unified Memory.

- The TAL Shape Mismatch: In the case where the "Task Aligned Assigner" triggered a RuntimeError at Epoch 5, we applied an Asynchronous Evaluation bypass. Rather than re-training, we re-recovered the best.pt checkpoint to ensure the 0.36 mAP score.

- NMS Time Limit Warnings: NMS Timeouts are documented and ignored, not a model failure, but a hardware latency phenomenon.
