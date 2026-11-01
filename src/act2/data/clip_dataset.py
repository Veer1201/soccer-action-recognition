import os
from torch.utils.data import Dataset

class UCF101ClipDataset(Dataset):
    def __init__(self, frames_root, clip_len=16, train=True):
        self.clip_len = clip_len
        self.train = train
        self.samples = []

        self.classes = []
        for name in sorted(os.listdir(frames_root)):
            if os.path.isdir(os.path.join(frames_root, name)):
                self.classes.append(name)

        for label in range(len(self.classes)):
            class_name = self.classes[label]
            class_dir = os.path.join(frames_root, class_name)

            for clip_name in sorted(os.listdir(class_dir)):
                clip_dir = os.path.join(class_dir, clip_name)
                if not os.path.isdir(clip_dir):
                    continue
                
                num_frames = 0
                for frame_file in os.listdir(clip_dir):
                    if frame_file.endswith(".jpg"):
                        num_frames += 1

                group_id = class_name + "_" + clip_name.split("_")[-2]
                self.samples.append((clip_dir, label, group_id, num_frames))

    def __len__(self):
        return len(self.samples)