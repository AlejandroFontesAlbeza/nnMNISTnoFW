from train import X_dev,Y_dev, forward_prop, get_predictions
import numpy as np
from matplotlib import pyplot as plt
import time

data = np.load('state_dict.npz')
W1 = data['W1']
b1 = data['b1']
W2 = data['W2']
b2 = data['b2']

def makePrediction(image, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, image)
    prediction = get_predictions(A2)
    return prediction

def testPrediction(index,W1,b1,W2,b2):
    current_image = X_dev[:,index,None]

    start = time.time()
    prediction = makePrediction(current_image,W1,b1,W2,b2)
    end = time.time()
    print('inference time:', end-start)


    label = Y_dev[index]
    print('prediction:', prediction)
    print('label:', label)
    
    
    current_image = current_image.reshape((28,28))*255
    plt.gray()
    plt.imshow(current_image)
    plt.savefig('label_inferenced.png')






def main_inference():
    index = 8
    testPrediction(index,W1,b1,W2,b2)


if __name__ == "__main__":
    main_inference()