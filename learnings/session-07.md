## Session 7 — 2026-10-07 / 10-28 / 10-29

### What we built
- Added group_id (e.g. Basketball_g01) to every sample
- Replaced random_split with a group-level split (Subset + seeded shuffle)
- Leakage check: groups in both = 0; split 82,112 train / 20,629 val frames
- Added .gitignore entries for dataset, __pycache__, .DS_Store
- Replaced the 20M-weight flatten classifier with global average pooling

### Concepts covered
- Data leakage: near-duplicate frames from the same video in train AND val
  make val accuracy look better than it really is
- Split by the unit that must not leak: UCF-101 group (same source video)
- Group split gives ~80/20 frames, not exact, because groups vary in size
- Honest result: val dropped from ~21% (leaky) to ~5% (group split)
- Chance level with 101 classes is ~1%
- Judge models by val accuracy; the train/val gap only explains why
- Model capacity: 99% of weights were in the classifier → easy memorization
- Global average pooling: [B, 64, 56, 56] → [B, 64]; removes location info,
  shrinks classifier from ~20M to ~6.5k weights
- Overfitting toolbox: GAP, augmentation, dropout, pretrained backbone

### New Python syntax learned
- str.split("_") and negative indexing ([-2])
- set(): removes duplicates, fast `in` checks; & for intersection
- Slicing: items[:n]
- `_` for unpacked values you don't use
- random.Random(seed).shuffle() for reproducible shuffling

### How it connects to the project
- Group-level splitting matters again in Act 4 (mAP would be inflated by leakage)
- GAP is used in ResNet / R3D-18, which we'll use next
- Datasets stay out of git; repo holds the code that builds them

### What confused me / what clicked
- I predicted val would go UP after fixing leakage; it went down, because
  leakage was inflating it
- A smaller gap isn't automatically better if val doesn't improve
- Building engineering logic: observe → ask why → plan in English → code → verify

### Results
- GAP run: Train acc ___ / Val acc ___ (fill in after training)
- Flatten: train 94% / val ~5% (epoch 2)
- GAP: train 33% / val ~4.3% (epoch 3), gap shrank but val didn't improve

### Next session
- Pretrained ResNet-18 (transfer learning) on frames
- Then augmentation and dropout on top
- Merge act2/2d-cnn-baseline into main via PR