import os
import av
import torch
from torchvision.io import write_jpeg


def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate)
    interval = max(1, int(video_fps / fps))

    saved_count = 0
    frame_count = 0

    for frame in container.decode(stream):
        if frame_count % interval == 0:
            img = frame.to_ndarray(format='rgb24')
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            write_jpeg(tensor, frame_path)
            saved_count += 1
        frame_count += 1

    container.close()
    return saved_count
