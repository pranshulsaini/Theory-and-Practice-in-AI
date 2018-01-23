import numpy as np
import numpy.matlib


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    xx = np.reshape(x, (x.shape[0], -1))   # converting (N,a,b,c) to (N, a*b*c). Now shape is NxD
    xx = np.hstack([xx, np.ones((xx.shape[0], 1))])  # for adding bias, Nx (D+1)
    ww = np.vstack([w, b]) # for adding bias, (D+1)xM
    out = xx.dot(ww)  #NxM
    
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    xx = np.reshape(x, (x.shape[0], -1))   # converting (N,a,b,c) to (N, a*b*c). Now shape is NxD
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    #pass
    # I have derived the formulas for linear classifier in my notebook
    db = sum(dout,0)
    
    #dx = dout * np.matlib.repmat(np.transpose(sum(w,1)), xx.shape[0], 1)
    dx = dout.dot(w.T)   
    # np.transpose(sum(w,1)) contains the derivatives of each dimension of x
    # repmat was used because derivates of x, which are in terms of w remains same for every x
    dw = xx.T.dot(dout)
    
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    dx = dx.reshape(*x.shape)
    
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    #pass
    out = x.clip(min=0)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    #pass
    out,_ =  relu_forward(x)
    out[out>0] = 1
    dx = dout * out
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU
  
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b) # fc_cache = (x, w, b)
    out, relu_cache = relu_forward(a)   # relu_cache = a 
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
