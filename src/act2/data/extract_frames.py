import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from src.act2.utils import extract_frames

def extract_all_frames(ucf_root, output_root, fps=1):
    classes = sorted(os.listdir(ucf_root))
    classes = [c for c in classes if os.path.isdir(os.path.join(ucf_root, c))]

    print(f"Found {len(classes)} classes")

    for class_name in classes:
        class_input = os.path.join(ucf_root, class_name)
        class_output = os.path.join(output_root, class_name)

        videos = [v for v in os.listdir(class_input) if v.endswith('.avi')]

        for video_name in videos:
            video_path = os.path.join(class_input, video_name)
            video_output = os.path.join(class_output, video_name.replace('.avi', ''))
            extract_frames(video_path, video_output, fps=fps)
        print(f"Done: {class_name} ({len(videos)} videos)")

if __name__ == "__main__":
    ucf_root = "src/act2/data/ucf101/UCF-101"
    output_root = "src/act2/data/ucf101/frames"
    extract_all_frames(ucf_root, output_root, fps=1)