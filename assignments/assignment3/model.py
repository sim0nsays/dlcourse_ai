import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, n_input, n_output, conv1_size, conv2_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        conv1_size, int - number of filters in the 1st conv layer
        conv2_size, int - number of filters in the 2nd conv layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        raise Exception("Not implemented!")

    def predict(self, X):
        # You can probably copy the code from previous assignment
        raise Exception("Not implemented!")

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        raise Exception("Not implemented!")

        return result
