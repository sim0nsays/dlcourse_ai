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
    # divide max value from all predictions to prevent float out of range
    predictions -= np.max(predictions)
    pred_exp = np.exp(predictions)

    if len(predictions.shape) > 1:
        res = pred_exp / np.sum(pred_exp, axis=1, keepdims=True)
    else:
        res = pred_exp / np.sum(pred_exp)
    return res


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
    # cross-entropy loss
    if type(target_index) is np.ndarray:
        res = - np.log(probs[np.arange(len(probs)), target_index])
    else:
        res = - np.log(probs[target_index])
    return res.mean()


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # Your final implementation shouldn't have any loops
    predictions = predictions.copy()
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    mask = np.zeros_like(predictions)
    if type(target_index) is np.ndarray:
        mask[np.arange(len(probs)), target_index] = 1
        d_preds = - (mask - softmax(predictions)) / mask.shape[0]
    else:
        mask[target_index] = 1
        d_preds = - (mask - softmax(predictions))
    return loss, d_preds


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = np.sum(W**2) * reg_strength
    grad = 2 * W * reg_strength
    return loss, grad


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
        self.X = None

    def forward(self, X):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        res = np.maximum(X, 0)
        self.X = X
        return res

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
        # Your final implementation shouldn't have any loops
        d_result = (self.X > 0) * d_out
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
        # Your final implementation shouldn't have any loops
        W, B = self.W.value, self.B.value
        self.X = Param(X)
        out = np.dot(X, W) + B
        return out

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
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        X, W = self.X.value, self.W.value
        dW, dX = np.dot(X.T, d_out), np.dot(d_out, W.T)
        dB = np.dot(np.ones((X.shape[0], 1)).T, d_out)

        self.W.grad += dW
        self.B.grad += dB

        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}
