from layers import (FullyConnectedLayer, ReLULayer, l2_regularization,
                    softmax_with_cross_entropy)


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
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def zero_grads(self):
        for param in self.params().values():
            param.grad[:] = 0.0

    def forward(self, X):
        out1 = self.layer1.forward(X)
        out2 = self.relu.forward(out1)
        preds = self.layer2.forward(out2)
        return preds

    def backward(self, d_preds):
        d_layer2 = self.layer2.backward(d_preds)
        d_relu = self.relu.backward(d_layer2)
        d_layer1 = self.layer1.backward(d_relu)
        return d_layer1

    def l2_regularization(self):
        for param in self.params().values():
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            param.grad += l2_grad
        return l2_loss

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        self.zero_grads()
        preds = self.forward(X)
        loss, d_preds = softmax_with_cross_entropy(preds, y)
        self.backward(d_preds)
        loss += self.l2_regularization()

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
            X, np array (test_samples, num_features)

        Returns:
            y_pred, np.array of int (test_samples)
        """
        preds = self.forward(X)
        return preds.argmax(axis=1)

    def params(self):
        result = {}
        result['W1'] = self.layer1.W
        result['B1'] = self.layer1.B
        result['W2'] = self.layer2.W
        result['B2'] = self.layer2.B

        return result
