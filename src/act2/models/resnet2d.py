import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes = 101, freeze_backbone = True):
    model = resnet18(weights = ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Linear(512, num_classes)
    return model

if __name__ == "__main__":
    model = build_resnet18()
    trainable = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()      # p.numel(): returns number of elements in a tensor
    print(f"Trainable params: {trainable:,}")