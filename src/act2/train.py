import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from src.act2.models.cnn2d import CNN2D
from src.act2.data import get_dataloader

from tqdm import tqdm

def train(frames_root, num_epochs=10, batch_size=32, lr=1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    model = CNN2D(num_classes=101).to(device)

    dataloader = get_dataloader(frames_root, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for  epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:4f} Acc: {epoch_acc:2f}%")

if __name__ == "__main__":
    train("src/act2/data/ucf101/frames", num_epochs=10)