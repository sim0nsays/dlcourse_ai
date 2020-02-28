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
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    return loss, dW


class LinearSoftmaxClassifier:
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, verbose=True, get_random_batch=True):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # randomly get batch number
            batch_num = np.random.randint(len(batches_indices))
            X_batches = X[batches_indices[batch_num]]
            target_index = y[batches_indices[batch_num]]
            loss, dW = linear_softmax(X_batches, self.W, target_index)
            l2_loss, l2_dW = l2_regularization(self.W, reg_strength=reg)
            loss += l2_loss
            loss_history.append(loss)

            dW += l2_dW
            self.W -= learning_rate * dW

            if verbose and (epoch == 0 or epoch == epochs - 1):
                print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        # Your final implementation shouldn't have any loops
        predictions = np.dot(X, self.W)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1)

        return y_pred
