## Session [9] — 2026-10-30 / 31

### What we built
- Train augmentation: RandomResizedCrop(scale 0.6–1.0), HorizontalFlip, ColorJitter
- `resnet18_finetune` option (all layers trainable, lr 1e-4)
- Optional dropout before the ResNet head (nn.Sequential)
- Best-checkpoint saving by val accuracy (+ checkpoints/ in .gitignore)
- Moved dataset code from `__init__.py` → `dataset.py`; formatted with Black
- PR act2/2d-cnn-baseline → main ✅
- SoccerNet access granted via Hugging Face NDA

### Results (val acc, frame-level UCF-101)
- Frozen 64.6% → frozen + aug 64.0% → fine-tune + aug 71.2% → + dropout 72.4%

### Concepts covered
- Augmentation barely helps a frozen backbone (ImageNet features already ignore flips/crops)
- Fine-tuning lets features adapt to actions → broke the ~64% ceiling
- Learning rate = step size, NOT speed. Speed depends on how many weights need gradients
- Dropout: randomly zero features in train mode only; no weights; survivors scaled ×2
- Val fluctuates ±2–4% between epochs → one run vs one run isn't proof
- Save the BEST epoch, not the last (epoch 2 dipped in both runs)
- Variables that remember across epochs go BEFORE the loop
- num_workers: parallel data loading. Didn't help here (GPU-bound + heat on fanless Air)
- MacBook Air thermal throttling: epochs slow down as the chip heats up

### New Python syntax learned
- Tuples: `Resize((224, 224))` vs `Resize(224, 224)` (2nd arg = interpolation!)
- `nn.Sequential`, `model.state_dict()`, `torch.save`
- `os.makedirs(..., exist_ok=True)`
- Packages: a folder with `__init__.py`; `__init__.py` re-exports names
- `python -c "..."` for quick tests; `grep -rl`; `black --diff`
- No spaces around `=` in function arguments

### How it connects to the project
- `checkpoints/resnet18_finetune_best.pt` → Act 3 server loads it (must rebuild the exact same architecture)
- 72.4% = frame-level baseline for the Act 4 comparison
- SoccerNet NDA: no public demo clips; ask about publishing weights

### What confused me / what clicked
- [your words]

### Next session
- Branch act2/3d-cnn
- GPU setup (Kaggle)
- Clip dataset: 16 frames per sample