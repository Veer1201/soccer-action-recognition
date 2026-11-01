import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from src.act2.utils import extract_frames, extract_frames_dense


def extract_all_frames(ucf_root, output_root, fps=1, dense=False, only_class=None):
    classes = sorted(os.listdir(ucf_root))
    classes = [c for c in classes if os.path.isdir(os.path.join(ucf_root, c))]

    if only_class is not None:
        classes = [only_class]

    print(f"Found {len(classes)} classes")
    total_frames = 0

    for class_name in classes:
        class_input = os.path.join(ucf_root, class_name)
        class_output = os.path.join(output_root, class_name)

        videos = [v for v in os.listdir(class_input) if v.endswith(".avi")]

        for video_name in videos:
            video_path = os.path.join(class_input, video_name)
            video_output = os.path.join(class_output, video_name.replace(".avi", ""))
            if dense:
                saved = extract_frames_dense(video_path, video_output)
            else:
                saved = extract_frames(video_path, video_output, fps=fps)
            total_frames += saved
        print(f"Done: {class_name} ({len(videos)} videos)")
    print(f"Total frames saved: {total_frames}")

if __name__ == "__main__":
    ucf_root = "src/act2/data/ucf101/UCF-101"
    output_root = "src/act2/data/ucf101/frames_128x171"
    extract_all_frames(ucf_root, output_root, dense=True)
