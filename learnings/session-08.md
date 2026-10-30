## Session [8] — 2026-10-29

### What we built
- `build_resnet18()`: pretrained ResNet-18, frozen backbone, new Linear(512, 101) head
- Model switch in train.py (`cnn2d` / `resnet18_frozen`)
- Separate train/val datasets so each split can have its own transform
- Frozen ResNet-18 baseline: **val 64.6%** (3 epochs, ~5–7 min/epoch on MPS)

### Concepts covered
- Transfer learning: reuse ImageNet features, replace only the head
- Freezing = `requires_grad = False` → no gradient → optimizer can't update it
- `.backward()` computes gradients; the optimizer uses them to update weights
- Only pass trainable params to the optimizer
- Trainable params for Linear(512, 101) = 512×101 + 101 = 51,813
- ResNet expects 224×224 + ImageNet normalization
- BatchNorm running stats are buffers, not params → keep the model in eval() when frozen
- Overfitting: train acc ↑, val acc flat, val loss ↑ (more confident wrong answers)
- Order matters: replace the head AFTER freezing, or it gets frozen too

### Bugs found
- **Val accuracy divided by the train total.** All earlier val numbers were ~4× too low.
  Corrected: CNN2D flatten ~20%, CNN2D GAP ~17%
- `os.listdir` order isn't guaranteed → wrapped in `sorted()`

### New Python syntax learned
- Generator expression: `sum(p.numel() for p in ... if p.requires_grad)`
- List comprehension: `[p for p in ... if p.requires_grad]`
- Default arguments: `def f(num_classes=101)`
- `if __name__ == "__main__":`
- `raise ValueError(...)`
- f-string formatting: `f"{x:,}"`
- `tensor.numel()`
- `git commit -am` (-a skips untracked files!)

### How it connects to the project
- Same pretrain → replace head → fine-tune pattern for R3D-18 and SoccerNet
- 64.6% frozen ResNet = frame-level baseline for the Act 4 comparison
- Single frames can't see motion → motivates the 3D CNN and two-stream steps
- Model-switch structure in train.py → easy to compare models in Act 4

### What confused me / what clicked
- [your words here]

### Next session
- Augmentation (fill in the `train=True` branch)
- Dropout before the head
- Unfreeze the backbone with a small LR (~1e-4)
- Save the best checkpoint (needed for Act 3)
- PR act2/2d-cnn-baseline → main