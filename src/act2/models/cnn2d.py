import torch
import torch.nn as nn

class CNN2D(nn.Module):
    def __init__(self, num_classes = 101):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
if __name__ == "__main__":
    torch.manual_seed(42)
    model = CNN2D(num_classes=101)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    print("output shape: ", output.shape)