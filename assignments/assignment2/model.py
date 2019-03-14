import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
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
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # After that, implement l2 regularization on all params
        # Hint: use self.params()
        raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        raise Exception("Not implemented!")

        return result
