import os
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

class UCF101Dataset(Dataset):
    def __init__(self, frames_root, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(frames_root))
        self.classes = [c for c in self.classes if os.path.isdir(os.path.join(frames_root, c))]

        self.class_to_idx = {c : i for i, c in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = os.path.join(frames_root, class_name)
            for clip_name in os.listdir(class_dir):
                clip_dir = os.path.join(class_dir, clip_name)
                group_tag = clip_name.split("_")[-2]
                group_id = f"{class_name}_{group_tag}"
                for frame_file in os.listdir(clip_dir):
                    if frame_file.endswith('.jpg'):
                        frame_path = os.path.join(clip_dir, frame_file)
                        label = self.class_to_idx[class_name]
                        self.samples.append((frame_path, label, group_id))

    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_path, label, _ = self.samples[idx]
        image = Image.open(frame_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
def get_dataloader(frames_root, batch_size=32, val_split=0.2, seed = 42):
        dataset = UCF101Dataset(frames_root, transform=get_transforms())

        group_ids = sorted(set(s[2] for s in dataset.samples))
        random.Random(seed).shuffle(group_ids)

        val_count = int(len(group_ids) * val_split)
        val_groups = set(group_ids[:val_count])

        train_idx = [i for i, s in enumerate(dataset.samples) if s[2] not in val_groups]
        val_idx = [i for i, s in enumerate(dataset.samples) if s[2] in val_groups]

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
