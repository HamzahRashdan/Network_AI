import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy(y, t):
    return -np.sum(t * np.log(y + 1e-7))

def init_network():
    network = {}
    network['W1'] = np.random.randn(784, 50) * 0.01
    network['b1'] = np.zeros(50)

    network['W2'] = np.random.randn(50, 100) * 0.01
    network['b2'] = np.zeros(100)

    network['W3'] = np.random.randn(100, 10) * 0.01
    network['b3'] = np.zeros(10)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y



def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        temp = x[idx]

        x[idx] = float(temp) + h
        fxh1 = f(x)

        x[idx] = float(temp) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = temp
        it.iternext()

    return grad
