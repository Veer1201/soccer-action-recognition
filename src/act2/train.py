import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.act2.models.cnn2d import CNN2D
from src.act2.models.resnet2d import build_resnet18
from src.act2.data import get_dataloader


def build_model(model_name, num_classes=101):
    if model_name == "cnn2d":
        return CNN2D(num_classes=num_classes)
    elif model_name == "resnet18_frozen":
        return build_resnet18(num_classes=num_classes, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def train(frames_root, model_name="cnn2d", num_epochs=5, batch_size=32, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training {model_name} on: {device}")

    model = build_model(model_name).to(device)
    frozen = model_name == "resnet18_frozen"

    train_loader, val_loader = get_dataloader(frames_root, batch_size)

    criterion = nn.CrossEntropyLoss()
    trainable_params = []
    for p in model.parameters():
        if p.requires_grad:
            trainable_params.append(p)

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    for epoch in range(num_epochs):
        if frozen:
            model.eval()
        else:
            model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total * 100

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% — "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    train("src/act2/data/ucf101/frames", model_name="resnet18_frozen", num_epochs=3)
