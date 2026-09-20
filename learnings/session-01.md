## Session 1 — 2026-09-15

### What we built
- Set up Python virtual environment and project structure
- Created src/act1/engine.py with a Value class skeleton
- Implemented __add__ and __mul__ on Value class
- Value objects now build a computation graph automatically on every operation

### Concepts covered
- What NumPy is and why it's faster than plain Python (C under the hood, vectorization)
- ndarray — n-dimensional array, core data structure of all ML
- .shape — returns (rows, columns) for 2D arrays, size of each dimension
- Vectorization — operating on whole arrays at once instead of looping
- What a computation graph is and why we need it for backpropagation
- How the Value class gives numbers memory by storing _prev and _op

### New Python syntax learned
- class and __init__ — blueprint for creating objects, constructor
- self — refers to the object itself, passed automatically
- Default arguments — _children=(), _op=''
- __repr__ — controls what prints when you print an object
- f-strings — f"Value(data={self.data})"
- set() — like a list but no duplicates, faster lookups
- Underscore prefix convention — _prev means internal/private
- __add__ and __mul__ — dunder methods Python calls for + and *

### How it connects to the project
- The Value class is the foundation of our entire autograd engine
- Every number in our neural network will be a Value object
- The computation graph built by __add__ and __mul__ is what 
  makes backpropagation possible in the next session
- In Act 2, PyTorch does exactly this internally — we'll understand 
  it because we built it ourselves

### What confused me / what clicked
- Initially thought print(a) would show a memory address — 
  clicked that __repr__ overrides that
- Indentation error taught me Python uses whitespace as syntax, 
  not just style
- The chain from a + b → __add__ → _children → _prev clicked fully

### Next session
- Add grad attribute to Value class
- Implement _backward function for addition and multiplication
- Implement .backward() to walk the full computation graph in reverse
- Complete the autograd engine