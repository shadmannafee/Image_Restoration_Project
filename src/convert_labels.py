import os
from ultralytics.data.converter import convert_coco

def main():
    json_dir = 'data/annotations'
    
    print("Target is instances_val2017.json")
    
    try:
        convert_coco(
            labels_dir=json_dir,
            use_segments=False,
            use_keypoints=False,
            cls91to80=True  # Important
        )
        print("\nConversion complete=========")
    except Exception as e:
        print(f"\nError during conversion: {e}")

if __name__ == "__main__":
    main()