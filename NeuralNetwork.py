import numpy as np
import Activations

activation_functions = {
    "sigmoid" : Activations.sigmoid,
    "unit-step" : Activations.unitStep,
}

activation_function_derivatives = {
    "sigmoid" : Activations.sigmoidDerivative
}

class NeuralNetwork:
    def __init__(self, num_inputs:int=-1, learning_rate:float=0.01) -> None:
        """
        Creates a neural network with num_inputs inputs

        Args:
            num_inputs (int): The number of inputs in the neural network

        Returns:
            Nothing.
        """
        if num_inputs <= 0:
            raise ValueError("Number of inputs must be greater than 0")

        self.learning_rate = learning_rate
        self.weight_matrices = []
        self.activation_functions = []
        self.neurons_in_layers = [num_inputs]
        self.activations = []

    def addLayer(self, neurons_in_layer:int=-1, activation_function:str=None) -> None:
        """
        Adds a layer to the neural network

        Args:
            neurons_in_layer (int): The number of neurons in the current layer
            activation_function (String): Name of the layer's activation function

        Returns:
            Nothing.
        """
        
        if neurons_in_layer <= 0:
            raise ValueError("Number of inputs must be greater than 0")
        elif not activation_function:
            raise ValueError("Must input type of activation function")
        
        # Add weight matrix that connects previous and current layer
        self.weight_matrices.append(np.random.uniform(low=-1.0, high=1.0, size=(neurons_in_layer, self.neurons_in_layers[-1])))
        # Append the activation function type of the layer to the activation functions list
        self.activation_functions.append(activation_function)
        # Add number of neurons in current layer to the list holding the number of neurons in all layers
        self.neurons_in_layers.append(neurons_in_layer)


    def forwardPropagation(self, input:np.ndarray) -> np.ndarray:
        """
        Performs forward propagation with the neural network

        Args:
            input (np.ndarray): The input vector that we pass to the network

        Returns:
            The output vector that the network produces
        """
        self.activations = []

        curr_activation = input
        for i in range(len(self.weight_matrices)):
            # Weight matrix for the current layer            
            layer_weights = self.weight_matrices[i]
            # Activation function for the current layer
            activation_function = activation_functions[self.activation_functions[i]]

            # Store the previous activation for backpropagation
            self.activations.append(curr_activation)

            # Find the activation produced by the current layer
            curr_activation = np.matmul(layer_weights, curr_activation)
            curr_activation = activation_function(curr_activation)

        return curr_activation
    
    def backwardPropagation(self, output:np.ndarray, desired:np.ndarray) -> None:
        """
        Performs backward propagation on the neural network

        Args:
            output (np.ndarray): The output vector produced by the neural network
            desired (np.ndarray): The desired output vector for the current input

        Returns:
            Nothing.
        """
        weight_updates = []

        # Find the last layer's delta and update its gradient 
        error_derivative = output - desired     # TODO: replace this with variable loss function (e.g. MSE, SIGNUM, ...)
        activation_function_derivative = activation_function_derivatives[self.activation_functions[-1]]
        prev_delta = error_derivative * activation_function_derivative(output)
        weight_updates.insert(0, self.learning_rate * np.matmul(prev_delta, self.activations[-1].T))

        # Find the remaining layers' deltas and gradients
        for i in range(len(self.weight_matrices) - 2, -1, -1):
            activation_function_derivative = activation_function_derivatives[self.activation_functions[i]]
            curr_delta = np.matmul(self.weight_matrices[i + 1].T, prev_delta) * activation_function_derivative(self.activations[i + 1])
            weight_updates.insert(0, self.learning_rate * np.matmul(curr_delta, self.activations[i].T))

            prev_delta = curr_delta

        # Update weight matrices using gradients
        for i in range(len(self.weight_matrices) - 1, -1, -1):
            self.weight_matrices[i] -= weight_updates[i]