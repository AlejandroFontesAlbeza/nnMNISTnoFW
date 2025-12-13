import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("train.csv.zip", compression='zip')

data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255



def init_params():
    W1 = np.random.rand(10,784)- 0.5
    b1 = np.random.rand(10,1)- 0.5

    W2 = np.random.rand(10,10)- 0.5
    b2 = np.random.rand(10,1)- 0.5

    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(0,Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() +1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1,A1,Z2,A2

def back_prop(Z1,A1,Z2,A2,W2,X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1,db1,dW2,db2


def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,lr):
    W1 = W1-lr*dW1
    b1 = b1-lr*db1
    W2 = W2-lr*dW2
    b2 = b2-lr*db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X,Y,epochs,lr):
    W1,b1,W2,b2 = init_params()
    for epoch in range(epochs):
        Z1,A1,Z2,A2 = forward_prop(W1,b1,W2,b2,X)
        dW1,db1,dW2,db2 = back_prop(Z1,A1,Z2,A2,W2,X,Y)
        W1,b1,W2,b2 = update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,lr)
        accuracy = get_accuracy(get_predictions(A2),Y)
        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, accuracy: {accuracy}')

    return W1,b1,W2,b2

def main():
    epochs,lr = (1000,0.1)
    print(f"Ejecutando entrenamiento de {epochs} iteraciones con un learning rate de {lr}")
    W1,b1,W2,b2 = gradient_descent(X_train,Y_train,epochs,lr)

    np.savez("state_dict.npz", W1 = W1, b1 = b1, W2 = W2, b2 = b2)


if __name__=="__main__":
    main()






