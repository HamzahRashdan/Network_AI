import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from network import (
    init_network,
    predict,
    cross_entropy,
    numerical_gradient
)

def train_once(x_train, t_train, x_test, t_test, title=""):
    iters_num = 1000
    batch_size = 100
    learning_rate = 0.1

    network = init_network()
    train_size = x_train.shape[0]

    loss_list = []

    for i in range(iters_num):

       
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

      
        def loss_W(_):
            y = predict(network, x_batch)
            return cross_entropy(y, t_batch)

        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, network['W1'])
        grads['b1'] = numerical_gradient(loss_W, network['b1'])
        grads['W2'] = numerical_gradient(loss_W, network['W2'])
        grads['b2'] = numerical_gradient(loss_W, network['b2'])
        grads['W3'] = numerical_gradient(loss_W, network['W3'])
        grads['b3'] = numerical_gradient(loss_W, network['b3'])

        
        for key in network.keys():
            network[key] -= learning_rate * grads[key]

        
        y = predict(network, x_batch)
        loss = cross_entropy(y, t_batch)
        loss_list.append(loss)

        if i % 20 == 0:
            print(f"{title} Iter {i}, Loss={loss}")

  
  
  
    def calc_acc(x, t):
        correct = 0
        for i in range(len(x)):
            y = predict(network, x[i])
            if np.argmax(y) == np.argmax(t[i]):
                correct += 1
        return correct / len(x)

    train_acc = calc_acc(x_train[:500], t_train[:500])
    test_acc = calc_acc(x_test[:500], t_test[:500])

    print(f"\n{title} Final Training accuracy = {train_acc}")
    print(f"{title} Final Test accuracy     = {test_acc}\n")

    return loss_list, train_acc, test_acc


if __name__ == "__main__":


    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    t_train = np.eye(10)[t_train]
    t_test = np.eye(10)[t_test]

   
    x_train_norm = x_train / 255.0
    x_test_norm = x_test / 255.0

    loss1, train_acc1, test_acc1 = train_once(
        x_train_norm, t_train,
        x_test_norm, t_test,
        title="With Normalization"
    )

    loss2, train_acc2, test_acc2 = train_once(
        x_train, t_train,
        x_test, t_test,
        title="Without Normalization"
    )

    plt.plot(loss1, label="With Normalization")
    plt.plot(loss2, label="Without Normalization")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Comparison")
    plt.show()

    
    
    
    
    
    
    
    
