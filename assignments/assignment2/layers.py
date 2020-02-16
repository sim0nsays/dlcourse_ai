import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
        predictions, np array, shape is either (N) or (batch_size, N) -
            classifier output

    Returns:
        probs, np array of the same shape as predictions -
            probability for every class, 0..1
    '''
    preds = predictions.copy()
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    preds -= preds.max(axis=1).reshape(-1, 1)
    return np.exp(preds) / np.sum(np.exp(preds), axis=1).reshape(-1, 1)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
        probs, np array, shape is either (N) or (batch_size, N) -
            probabilities for every class
        target_index: np array of int, shape is (1) or (batch_size) -
            index of the true class for given sample(s)

    Returns:
        loss: single value
    '''
    if isinstance(target_index, int):
        target_index = np.array([target_index])
    return np.mean(
        -np.log(probs[range(len(probs)), target_index.reshape(1, -1)])
    )


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
        W, np array - weights
        reg_strength - float value

    Returns:
        loss, single value - l2 regularization loss
        gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W**2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
        predictions, np array, shape is either (N) or (batch_size, N) -
            classifier output
        target_index: np array of int, shape is (1) or (batch_size) -
            index of the true class for given sample(s)

    Returns:
        loss, single value - cross-entropy loss
        dprediction, np array same shape as predictions -
        gradient of predictions by loss value
    """
    if isinstance(target_index, int):
        target_index = np.array([target_index])
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    dprediction[range(len(probs)), target_index.reshape(1, -1)] -= 1
    dprediction /= len(probs)

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        out = np.maximum(0, X)
        self.grad = 1 * (out > 0)
        return out

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
            d_out, np array (batch_size, num_features) - gradient
                of loss function with respect to output

        Returns:
            d_result: np array (batch_size, num_features) - gradient
                with respect to input
        """
        d_result = d_out * self.grad
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return self.X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
            d_out, np array (batch_size, n_output) - gradient
                of loss function with respect to output

        Returns:
            d_result: np array (batch_size, n_input) - gradient
                with respect to input
        """
        d_input = d_out @ self.W.value.T
        self.W.grad = self.X.T @ d_out
        self.B.grad = d_out.sum(axis=0).reshape(self.B.value.shape)

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
