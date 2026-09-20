from engine import Value
from neuron import Neuron

# simple pattern: positive sum -> 1, negative sum -> -1
xs = [(2.0, 3.0), (3.0, 1.0), (-1.0, -2.0), (-2.0, -3.0)]
ys = [1.0, 1.0, -1.0, -1.0]

n = Neuron(2)

for i in range(5000):
    ypred = []
    for x in xs:
        output = n(x)
        ypred.append(output)

    for weight in n.weights:
        weight.grad = 0.0
    n.bias.grad = 0.0

    loss = Value(0.0)
    for ypredi, ysi in zip(ypred, ys):
        loss = loss + (ypredi - ysi) ** 2

    loss.backward()
    if i == 0:
        print(f"grads: w0={n.weights[0].grad:.4f}, w1={n.weights[1].grad:.4f}, b={n.bias.grad:.4f}")

    learning_rate = 0.0001
    for weight in n.weights:
        weight.data = weight.data - learning_rate * weight.grad
    n.bias.data = n.bias.data - learning_rate * n.bias.grad

    if i % 100 == 0:
        print(f"iteration {i}, loss: {loss.data:.4f}")


print("\n Final Predictions:")
for x, y in zip(xs, ys):
    pred = n(x)
    print(f"input: {x}, target: {y}, predicted: {pred.data:.4f}")
