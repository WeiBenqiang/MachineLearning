import numpy as np
import matplotlib.pyplot as plt
"""
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

x = np.linspace(-10, 10, 1000)
y = sigmoid(x)
plt.plot(x,y)
plt.show()
"""
sizes = [3, 4, 5]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(biases)
weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
print(weights[1])
print("--------------------------")
nabla_b = [np.zeros(b.shape) for b in biases]
print(nabla_b)
nabla_w = [np.zeros(w.shape) for w in weights]
for w in weights:
    print(w.shape)
print("------------------")
print(nabla_w)