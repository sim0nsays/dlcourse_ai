import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test][i_train] = (np.abs(X[i_test] - self.train_X[i_train])).sum()

            return dists

    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = (np.abs(X[i_test] - self.train_X)).sum(axis=1)

        return dists

    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dimension = self.train_X.shape[1]
        # Using float32 to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.float32)
        dists = (np.abs(X[:, np.newaxis, :] -
                        self.train_X[np.newaxis, :, :])).sum(axis=2)
        # dists = (np.abs(X.reshape(num_test, 1, X.shape[1]) -
        #                 self.train_X.reshape((1, num_train, X.shape[1])))).sum(axis=2)
        # template = np.zeros((num_test, num_train, dimension))
        # dists = (np.abs(template + X.reshape(num_test, 1, X.shape[1]) -
        #                 self.train_X.reshape((1, num_train, X.shape[1])))).sum(axis=2)
        # dists = np.sqrt(
        #     - 2 * X.dot(self.train_X.T) + (X**2).sum(axis=1, keepdims=True) +
        #     (self.train_X**2).T.sum(axis=0, keepdims=True)
        # )  # L2

        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            pred[i] = (self.train_y[np.argsort(dists[i])[:self.k]].mean() > 0.5)

        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)

        def count_item(arr, item):
            return np.sum(arr == item)

        for i in range(num_test):
            # TODO: Implement choosing best class based on k
            # nearest training samples
            choose = self.train_y[np.argsort(dists[i])[:self.k]]
            count = {}
            for item in np.unique(choose):
                count[item] = count_item(choose, item)
            count_sorted = sorted(count.items(), key=lambda x: x[1], reverse=True)
            pred[i] = count_sorted[0][0]

        return pred
