## Session 2 — 2026-09-15

### What we built
- Completed the backward pass for addition and multiplication
- Added grad attribute to every Value object
- Implemented _backward functions for + and * using chain rule
- Built the .backward() method with topological sort
- Handled plain numbers in add and multiply operations
- Tested the full engine with a multi-operation computation graph

### Concepts covered
- Gradients — how much the output changes when a value changes
- Chain rule — de/da = de/dc * dc/da, chaining gradients backwards
- Why gradients accumulate (+=) instead of overwrite (=)
- Topological sort — ordering nodes so parents are processed before children
- Why reversed(topo) gives us the right backward order
- Loss function vs backprop — loss measures error, backprop computes gradients
- Activation functions — why neural networks need them (non-linearity)

### New Python syntax learned
- lambda — anonymous function, used as placeholder for _backward
- Nested functions / closures — _backward defined inside __add__ 
  and can access self, other, out from outer scope
- Passing a function vs calling it — _backward vs _backward()
- Ternary operator — x if condition else y
- isinstance(obj, Class) — checks if an object is an instance of a class
- Recursive functions — build_topo calls itself to walk the graph

### How it connects to the project
- This is exactly what PyTorch's autograd does internally in Act 2
- Every .backward() call on our loss function in Act 2 runs this 
  same algorithm under the hood
- Understanding this means we'll never be confused by PyTorch gradients

### What confused me / what clicked
- Gradients were confusing at first but clicked after tracing through
  the chain rule manually with concrete numbers
- The difference between passing a function and calling it was a 
  key distinction

### Next session
- Implement ** power operation
- Implement tanh activation function
- Use the completed engine to train a tiny neural network