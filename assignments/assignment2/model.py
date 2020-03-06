import sys
import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # init NN layers
        self.linear1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.linear2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # get all layers parameters
        params = self.params()
        W1, B1 = params['W1'], params['B1']
        W2, B2 = params['W2'], params['B2']

        # clean gradients
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)

        # Compute loss and fill param gradients
        # by running forward and backward passes through the model

        out1 = self.linear1.forward(X)
        relu_out = self.relu.forward(out1)
        out2 = self.linear2.forward(relu_out)

        loss, d_preds = softmax_with_cross_entropy(out2, y)

        d_out2 = self.linear2.backward(d_preds)
        d_relu_out = self.relu.backward(d_out2)
        d_out1 = self.linear1.backward(d_relu_out)

        # l2 regularization
        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)

        # calculate l2 loss across all params
        l2_reg_loss = l2_W1_loss + l2_B1_loss + l2_W2_loss + l2_B2_loss
        # calculate final loss
        loss += l2_reg_loss

        # update gradients
        W1.grad += l2_W1_grad
        B1.grad += l2_B1_grad
        W2.grad += l2_W2_grad
        B2.grad += l2_B2_grad
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        out1 = self.linear1.forward(X)
        relu_out = self.relu.forward(out1)
        predictions = self.linear2.forward(relu_out)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1)
        return y_pred

    def params(self):
        linear1_params, linear2_params = self.linear1.params(), self.linear2.params()
        result = {
            'W1': linear1_params['W'], 'B1': linear1_params['B'],
            'W2': linear2_params['W'], 'B2': linear2_params['B']
        }
        return result
