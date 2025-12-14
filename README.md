# nnMNISTnoFW

<p align = "center">
    <img src = "resources/MNIST.png" alt = "Gradient Descent w and b ideals" width = "350"/>
</p>

Neural network implementation from scratch

Neural network implementation from scratch (no frameworks) for handwritten digit classification using the MNIST dataset.

This project focuses on understanding the **core fundamentals of neural networks**, including forward propagation, backpropagation, and gradient descent, without relying on high-level deep learning frameworks such as PyTorch or TensorFlow.


###  Project Motivation

The main goal of this project is **educational**:  
to deeply understand how a neural network works internally before using modern frameworks.

By implementing everything manually using only `numpy`, `pandas` and `matplotlib`, this project helps demystify:
- How neurons compute outputs
- How errors are propagated backward
- How weights are updated during training
- How inference works once the model is trained

This approach provides a strong foundation for later working with frameworks like PyTorch in more complex production environments.

---

###  Model Architecture

The model is a **simple fully connected neural network** with:

- Input layer: 28 × 28 pixels (flattened MNIST images)
- Hidden layer: Linear layer
- Output layer: 10 neurons (digits 0–9)
- Activation functions: implemented manually
- Loss function: implemented manually
- Optimization: gradient descent


###  Training 

- The Network was trained during 1000 epochs (**duration: 5min**)
- Training is performed from scratch
- After training the weights and biases are saved with `numpy` as .npz
- Final accuracy: **84%**


###  Inference 

- Loads the saved weights
- Runs predictions on test images never seen during training
- Outputs the predicted digit and saves de inferenced image

---

###  Results & Observations 

An accuracy of ~84% is a solid result given:

- The simplicity of the model

- The absence of modern techniques (regularization, batch normalization, CNNs, etc.)

- The fact that everything is implemented manually

However:

- The model can still fail on ambiguous or poorly written digits

- Some misclassifications are expected due to the linear nature of the network

- Increasing training epochs, tuning hyperparameters, or adding non-linear layers could improve performance

Despite these limitations, the project successfully demonstrates the core mechanics of neural networks and provides a strong conceptual foundation.


### Improvements

- Add more hidden layers or neurons

- Experiment with different **activation** functions

- Implement regularization techniques

- Replace the linear model with a convolutional neural network (**CNN**)

- Reimplement the same model using **PyTorch** for comparison

---

### References & Learning Resources

- Building a Neural Network from scratch - Medium: https://medium.com/@murmuarpan530/building-a-neural-network-from-scratch-in-python-e78203471870

- Building a neural network FROM SCRATCH - Samson Zhang: https://www.youtube.com/watch?v=w8yWXqWQYmU