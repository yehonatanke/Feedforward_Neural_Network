"""
predict.py: Contains functions for making predictions using the trained model.

Functions:
    predict(X, parameters): Makes predictions using the trained model.
"""

import numpy as np
from model import forward_propagation

def predict(X, parameters):
    """
    Makes predictions using the trained model.

    Args:
        X (numpy.ndarray): Input data.
        parameters (tuple): Trained parameters of the model.

    Returns:
        numpy.ndarray: Predicted labels.
    """
    _, a2 = forward_propagation(X, parameters)
    return np.argmax(a2, axis=1)
