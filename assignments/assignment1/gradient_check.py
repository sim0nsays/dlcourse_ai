import numpy as np


def numeric_grad_array(f, x, h):
    """
    calculating numerical differentiation 2-point formula: (f(x+h) - f(x-h))/2h
    source: https://en.wikipedia.org/wiki/Numerical_differentiation

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      h: small change in x to compute numerical gradient

    Return:
      numpy.nd.array of numerical gradient
    """
    dx = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x_plus_h, x_minus_h = x.copy(), x.copy()
        x_plus_h[ix] += h
        x_minus_h[ix] -= h
        dx[ix] = (f(x_plus_h)[0] - f(x_minus_h)[0]) / (2 * h)
        it.iternext()
    return dx


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    numeric_grad = numeric_grad_array(f, x, h=delta)

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = numeric_grad[ix]

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix,
                                                                                      numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
