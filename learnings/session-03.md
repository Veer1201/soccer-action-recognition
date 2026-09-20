## Session 3 — 2026-09-19

### What we built
- Power operation (__pow__)
- Subtraction (__sub__) built from add and multiply
- Division (__truediv__) built from multiply and power
- tanh activation function with correct gradient

### Concepts covered
- Why activation functions exist — without them networks stay linear
- tanh squishes any number to between -1 and 1
- Derivative of tanh is 1 - tanh(x)² — elegant and cheap to compute
- Subtraction and division don't need new backward logic — 
  reuse existing operations, chain rule handles the rest
- __truediv__ vs __div__ — Python 3 uses __truediv__ for /

### New Python syntax learned
- Nothing new this session — consolidation session

### What confused me / what clicked
- tanh formula and its gradient were shaky at first
- Clicked that t = tanh(x) computed in forward pass gets reused 
  directly in _backward — no extra computation needed

### Next session
- Build a single neuron using the completed engine
- output = tanh(w1*x1 + w2*x2 + b)
- Watch it actually learn from training data