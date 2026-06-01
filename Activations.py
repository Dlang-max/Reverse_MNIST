import numpy as np

def sigmoid(v:np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function to the vector of induced local fields
    passed as input

    Args:
        v (np.ndarray): vector of induced local fields

    Returns:
        Vector of the activations produced by applying the sigmoid function to the input
    """
    return 1 / (1 + np.exp(-v))

def sigmoidDerivative(sigmoid:np.ndarray) -> np.ndarray:
    """
    Returns the derivative of the inputted vector sigmoid activations

    Args:
        sigmoid (np.ndarray): vector of sigmoid activations

    Returns:
        Vector of derivatives of inputted sigmoid activations         
    """
    return sigmoid * (1 - sigmoid)


def unitStep(v:np.ndarray) -> np.ndarray:
    """
    Applies the unit step activation function to the vector of induced local fields
    passed as input

    Args:
        v (np.ndarray): vector of induced local fields

    Returns:
        Vector of the activations produced by applying the unit step function to the input
    """
    return (v > 0).astype(np.float64)


def unitStepDerivative(unitStep:np.ndarray) -> np.ndarray:
    """
    Returns the derivative of the inputted unit step activations

    Args:
        unitStep (np.ndarray): vector of unit step activations

    Returns:
        Vector of derivatives of inputted unit step activations         
    """
    return np.zeros_like(unitStep)