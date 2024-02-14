# Feedforward Neural Network (Multilayer Perceptron)

## Overview
A Python implementation of a feedforward neural network, also known as a multilayer perceptron (MLP). The neural network is trained using gradient descent with backpropagation and includes functionalities for splitting the dataset, training the model, and evaluating its performance.

## Features
- **Feedforward Neural Network**: Implements a basic feedforward neural network architecture with one hidden layer.
- **Gradient Descent Optimization**: Utilizes gradient descent for optimizing the model parameters (weights and biases).
- **Early Stopping**: Implements early stopping based on validation loss to prevent overfitting.
- **Documentation**: Includes detailed documentation for each function to aid understanding and usage.

## Usage
To train the neural network, use `train_model` function in the `train.py` file.
A basic example:

```python
from train import train_model, split_data
from model import forward_propagation, relu, softmax, cross_entropy_loss

# Load and preprocess the dataset
X, y = load_data()
X_train, X_val, y_train, y_val = split_data(X, y)

# Define model parameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))
learning_rate = 0.01
epochs = 100
patience = 5

# Train the model
best_params = train_model(X_train, y_train, X_val, y_val,
                           input_size, hidden_size, output_size,
                           learning_rate, epochs, patience)
```

## Neural Network Architecture
The feedforward neural network consists of the following layers:

1. **Input Layer**: Accepts input features.
2. **Hidden Layer**: Contains hidden units with weights and biases.
3. **Output Layer**: Produces output probabilities.

### Algebraic Representation of Network Components:

1. **Input to Hidden Layer**:

   $Z^{[1]} = X \cdot W^{[1]} + b^{[1]}$
   $A^{[1]} = \text{ReLU}(Z^{[1]})$

2. **Hidden to Output Layer**:

   $Z^{[2]} = A^{[1]} \cdot W^{[2]} + b^{[2]}$
   $A^{[2]} = \text{Softmax}(Z^{[2]})$

3. **Loss Computation**:

   $\text{Loss} = -\frac{1}{m} \sum_{i=1}^m  \\sum_{k=1}^K y_{i,k} \log(a_{i,k}^{[2]})$

4. **Backpropagation**:

   $\frac{\partial \text{Loss}}{\partial Z^{[2]}} = A^{[2]} - Y$
   
   $\frac{\partial \text{Loss}}{\partial W^{[2]}} = A^{[1]T} \cdot \frac{\partial \text{Loss}}{\partial Z^{[2]}}$
   
   $\frac{\partial \text{Loss}}{\partial b^{[2]}} = \sum_{i=1}^{m} \frac{\partial \text{Loss}}{\partial Z^{[2]}}$
   
   $\frac{\partial \text{Loss}}{\partial A^{[1]}} = \frac{\partial \text{Loss}}{\partial Z^{[2]}} \cdot W^{[2]T}$
   
   $\frac{\partial \text{Loss}}{\partial Z^{[1]}} = \frac{\partial \text{Loss}}{\partial A^{[1]}} \cdot \text{ReLU'}(Z^{[1]})$
   
   $\frac{\partial \text{Loss}}{\partial W^{[1]}} = X^T \cdot \frac{\partial \text{Loss}}{\partial Z^{[1]}}$
   
   $\frac{\partial \text{Loss}}{\partial b^{[1]}} = \sum_{i=1}^{m} \frac{\partial \text{Loss}}{\partial Z^{[1]}}$
