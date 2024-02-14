"""
train.py: Contains functions for training the neural network model.

Functions:
    cross_entropy_loss(y_pred, y_true): Computes the cross-entropy loss.
    loss_gradient(y_pred, y_true): Computes the gradient of the loss function.
    backward_propagation(X, y, a1, a2, parameters): Performs backward propagation.
    gradient_descent(parameters, gradients, learning_rate): Updates parameters using gradient descent.
    train_model(X, y, input_size, hidden_size, output_size, learning_rate, epochs): Trains the neural network model.
"""

import numpy as np
from model import forward_propagation, relu, softmax

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

def loss_gradient(y_pred, y_true):
    """
    Computes the gradient of the loss function.

    Args:
        y_pred (numpy.ndarray): Predicted probabilities.
        y_true (numpy.ndarray): True labels.

    Returns:
        numpy.ndarray: Gradient of the loss.
    """
    m = y_true.shape[0]
    grad = y_pred.copy()
    grad[range(m), y_true] -= 1
    grad /= m
    return grad

def backward_propagation(X, y, a1, a2, parameters):
    """
    Performs backward propagation.

    Args:
        X (numpy.ndarray): Input data.
        y (numpy.ndarray): True labels.
        a1 (numpy.ndarray): Output of the hidden layer.
        a2 (numpy.ndarray): Output of the output layer.
        parameters (tuple): Tuple containing weights and biases.

    Returns:
        tuple: Gradients of the parameters.
    """
    W1, b1, W2, b2 = parameters
    m = X.shape[0]
    dz2 = loss_gradient(a2, y)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * (a1 > 0)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def gradient_descent(parameters, gradients, learning_rate):
    """
    Updates parameters using gradient descent.

    Args:
        parameters (tuple): Tuple containing weights and biases.
        gradients (tuple): Gradients of the parameters.
        learning_rate (float): Learning rate.

    Returns:
        tuple: Updated parameters.
    """
    W1, b1, W2, b2 = parameters
    dW1, db1, dW2, db2 = gradients
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train_model(X_train, y_train, X_val, y_val, input_size, hidden_size, output_size, learning_rate, epochs, patience=5):
    """
    Trains the neural network model.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation labels.
        input_size (int): Number of input features.
        hidden_size (int): Number of units in the hidden layer.
        output_size (int): Number of output units.
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of epochs.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping.

    Returns:
        tuple: Trained parameters.
    """
    best_loss = np.inf
    patience_count = 0
    best_params = None
    
    np.random.seed(0)
    W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
    b2 = np.zeros((1, output_size))
    parameters = (W1, b1, W2, b2)
    
    for epoch in range(epochs):
        # Forward propagation
        a1, a2 = forward_propagation(X_train, parameters)
        
        # Compute loss
        loss = cross_entropy_loss(a2, y_train)
        
        # Backward propagation
        dz2 = a2 - y_train
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (a1 > 0)
        dW1 = np.dot(X_train.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Gradient descent optimization
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        # Compute validation loss
        _, a2_val = forward_propagation(X_val, parameters)
        val_loss = cross_entropy_loss(a2_val, y_val)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            best_params = parameters
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break
    
    return best_params

def split_train_validation(X_train, y_train, validation_size=0.1):
    """
    Splits the training data into training and validation sets.

    Args:
        X_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        validation_size (float): Percentage of data to use for validation.

    Returns:
        tuple: Training data and labels, validation data and labels.
    """
    return train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

def augment_data(X_train, y_train):
    from imgaug.augmenters import SomeAugmenter
    # Define augmentation transformations
    augmenter = SomeAugmenter()
    # Apply augmentation to X_train
    X_augmented = augmenter(images=X_train)
    return X_augmented, y_train

def forward_propagation_with_dropout(X, parameters, keep_prob=0.8):
    W1, b1, W2, b2 = parameters
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    # Apply dropout to hidden layer
    dropout_mask = np.random.rand(*a1.shape) < keep_prob
    a1 *= dropout_mask / keep_prob
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    return a1, a2

def exponential_decay(initial_lr, epoch, decay_rate):
    """
    Computes the learning rate with exponential decay.

    Args:
        initial_lr (float): Initial learning rate.
        epoch (int): Current epoch.
        decay_rate (float): Decay rate.

    Returns:
        float: Updated learning rate.
    """
    return initial_lr * np.exp(-decay_rate * epoch)

def l2_regularization(W1, W2, lambda_):
    """
    Computes L2 regularization penalty.

    Args:
        W1 (numpy.ndarray): Weights of the first layer.
        W2 (numpy.ndarray): Weights of the second layer.
        lambda_ (float): Regularization parameter.

    Returns:
        float: L2 regularization penalty.
    """
    return 0.5 * lambda_ * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

