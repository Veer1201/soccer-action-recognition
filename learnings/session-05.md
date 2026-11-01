## Session 5 — 2026-09-20

### What we built
- Installed PyTorch 2.14 and torchvision, confirmed MPS available on MacBook Air
- Scaffolded act2 folder structure (data/, models/, train.py, utils.py)
- Built TinyNet — minimal PyTorch model with one nn.Linear layer
- Built CNN2D — two conv layers, ReLU, MaxPool, flatten, classifier
- Verified CNN2D output shape [1, 101] with a fake input tensor

### Concepts covered
- Tensors — multi-dimensional arrays, the core data structure in PyTorch
- .shape — tells you how many elements exist along each dimension
- nn.Module — base class for every PyTorch model, requires __init__ and forward
- nn.Linear — fully connected layer, takes in_features and out_features
- Convolutional filter — small window (3x3) that slides across image learning patterns
- in_channels/out_channels — input and output feature maps per conv layer
- padding — adds zeros around image border so spatial size is preserved
- ReLU — activation function, zeros out negatives, keeps positives
- MaxPool — halves height and width by keeping max value in each 2x2 region
- x.view() — reshapes tensor without changing values, bridges conv layers to linear
- random seed — torch.manual_seed(42) makes random initialization reproducible
- MPS device — Apple Silicon GPU, faster than CPU for training

### New Python syntax learned
- if __name__ == "__main__": — block that only runs when file is executed directly
- torch.randn() — creates tensor of random numbers with given shape

### How it connects to the project
- CNN2D is our Act 2 baseline model for classifying actions in single video frames
- The [batch, channels, height, width] shape convention applies to every model in Act 2/3/4
- MPS device flag we'll add to every training script for faster training

### What confused me / what clicked
- == vs = bug in __init__ — comparison vs assignment, same mistake as Act 1
- Tracing shapes through the network took a few tries — kept dropping the channels dimension
- x.view() clicked once I understood nn.Linear needs 1D input per image

### Next session
- Build the training loop for CNN2D
- Load UCF-101 dataset using torchvision
- Train on MPS device
- Track loss and accuracy per epoch