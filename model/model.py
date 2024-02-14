import numpy as np

def relu(x):
    """
    Computes the rectified linear unit (ReLU) activation function.

    Args:
        x (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Output after applying ReLU activation.
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Computes the softmax activation function.

    Args:
        x (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Output after applying softmax activation.
    """
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def forward_propagation(X, parameters):
    """
    Performs forward propagation through the neural network layers.

    Args:
        X (numpy.ndarray): Input features.
        parameters (tuple): Model parameters (weights and biases).

    Returns:
        tuple: Outputs of the hidden layer and output layer.
    """
    W1, b1, W2, b2 = parameters
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a1, a2

def cross_entropy_loss(y_pred, y_true):
    """
    Computes the cross-entropy loss.

    Args:
        y_pred (numpy.ndarray): Predicted probabilities.
        y_true (numpy.ndarray): True labels.

    Returns:
        float: Cross-entropy loss.
    """
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    loss = np.sum(log_likelihood) / m
    return loss
