import cv2
import os
import json
import numpy as np

def main():
    # Paths
    original_dir = "data/val2017"
    restored_dir = "data/task4_train/images/train"
    metadata_path = "data/blur_metadata.json"
    output_path = "runs/qualitative_comparison_final.png"
    
    if not os.path.exists(metadata_path):
        print("Error: Metadata not found.")
        return

    with open(metadata_path, 'r') as f:
        data = json.load(f)
    metadata = data.get('transformations', {})

    selected = {'light': None, 'medium': None, 'heavy': None}
    for img_name, info in metadata.items():
        size = info['size']
        if size <= 10 and selected['light'] is None: selected['light'] = img_name
        elif 10 < size <= 20 and selected['medium'] is None: selected['medium'] = img_name
        elif size > 20 and selected['heavy'] is None: selected['heavy'] = img_name
        if all(selected.values()): break

    # Create Header Row (Original | Blurred | Restored)
    # Each image is 300px, so header is 900px wide. 
    # Adding 100px on the left for the Tier labels.
    header = np.zeros((60, 1000, 3), dtype=np.uint8)
    cv2.putText(header, "Original", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(header, "Blurred", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(header, "Restored", (780, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    rows = [header]

    for level, img_name in selected.items():
        # Load and process
        orig = cv2.imread(os.path.join(original_dir, img_name))
        rest = cv2.imread(os.path.join(restored_dir, img_name))
        
        info = metadata[img_name]
        size, angle = info['size'], info['angle']
        k = np.zeros((size, size))
        k[int((size-1)/2), :] = np.ones(size)
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        k = cv2.warpAffine(k, M, (size, size))
        k = k / np.sum(k)
        blurred = cv2.filter2D(orig, -1, k)

        # Resize images to 300x300
        orig = cv2.resize(orig, (300, 300))
        blurred = cv2.resize(blurred, (300, 300))
        rest = cv2.resize(rest, (300, 300))

        sidebar = np.zeros((300, 100, 3), dtype=np.uint8)
        cv2.putText(sidebar, level.upper(), (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        row = np.hstack((sidebar, orig, blurred, rest))
        rows.append(row)

    # Final Grid
    grid = np.vstack(rows)
    
    os.makedirs('runs', exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"Success! Final comparison saved to: {output_path}")

if __name__ == "__main__":
    main()