## Session 6 — 2026-09-21

### What we built
- Downloaded and extracted UCF-101 (13,320 videos, 101 classes)
- Frame extraction at 1 fps with PyAV (~102.7k frames)
- UCF101Dataset (Dataset subclass) with __len__ and __getitem__
- Transform pipeline: Resize(224) → ToTensor → Normalize (ImageNet stats)
- DataLoader with batching and shuffling
- Training loop with CrossEntropyLoss + Adam on MPS, tqdm progress bar
- Train/val evaluation with model.eval() and torch.no_grad()

### Concepts covered
- Why images must be resized: the classifier input size was hardcoded
- ToTensor: PIL → tensor AND rescales [0, 255] → [0.0, 1.0]
- Normalize: shifts each channel to mean ~0, std ~1; must come after ToTensor
- Dataset vs DataLoader: storage + one-sample access vs batching/shuffling/loading
- Lazy loading: store paths in __init__, load pixels in __getitem__
- Shuffle train (avoid class-ordered batches), don't shuffle val (reproducible)
- CrossEntropyLoss for classification, Adam optimizer
- argmax(dim=1) picks the highest-scoring class per image
- model.train() / model.eval(); torch.no_grad() skips gradient tracking
- Overfitting: train acc rises while val acc stalls and val loss rises

### New Python syntax learned
- List comprehensions with filters: [c for c in items if cond]
- Dict comprehensions: {c: i for i, c in enumerate(classes)}
- enumerate(): index + value while looping
- f-strings, including padding like {n:04d}
- Dunder methods: __len__, __getitem__
- while True / break
- Unpacking multiple return values: ret, frame = cap.read()

### How it connects to the project
- This Dataset/DataLoader pattern is reused for 3D CNN clips and SoccerNet
- Same frame extraction feeds the Act 3 inference server

### What confused me / what clicked
- OpenCV silently read only 1 frame per clip on Mac → switched to PyAV
- torchvision 0.29 removed read_video
- Bug: val metrics used training variables, so train and val matched exactly
- Two classes vs 101 classes confusion with the extraction script

### Next session
- Fix overfitting, re-check how train/val are split