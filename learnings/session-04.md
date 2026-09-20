## Session 4 — 2026-09-19

### What we built
- Neuron class with random weight initialization
- Forward pass: weighted sum of inputs + bias, passed through tanh
- Full training loop with MSE loss and gradient descent
- Single neuron converging to loss ~0.02 on a simple classification task

### Concepts covered
- What a neuron is: weighted inputs + bias + activation function
- MSE loss — measures how wrong predictions are by squaring the error
- Gradient descent — nudge weights in direction that reduces loss
- Why we zero gradients before each backward pass
- Why weights update uses .data directly — to avoid creating new graph nodes
- Learning rate — controls step size, too high = diverge, too low = slow
- Gradient explosion — weights growing unboundedly when lr is too high

### New Python syntax learned
- zip — loop over two lists simultaneously, pairing elements together
- for wi, xi in zip(a, b) — unpacking pairs directly in the loop header

### What confused me / what clicked
- The sequential order of training loop clicked: forward → zero grads 
  → backward → update weights. Getting the order wrong broke everything.
- The = vs += bug in tanh _backward taught me how one character can 
  silently break the entire gradient computation
- Operator precedence bug: (t**2) * out.grad vs (1 - t**2) * out.grad
  — brackets matter

### Next session
- Act 1 is complete
- Move to Act 2: PyTorch basics
- Build a 2D CNN baseline on individual frames
- Start with UCF-101 dataset