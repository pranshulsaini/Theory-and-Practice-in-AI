import numpy as np

from AI.layers import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. I assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    
    The architecure is affine - relu - affine - softmax.
  
    This class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
  
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initializing a new network.
    
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # Initializing the weights and biases of the two-layer net. Weights        #
        # are initialized from a Gaussian with standard deviation equal to         #
        # weight_scale, and biases are initialized to zero. All weights and        #
        # biases are stored in the dictionary self.params, with first layer        #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        #print(self.params['W1'].std())
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        #print(self.params['W2'].std())
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Computing loss and gradient for a minibatch of data.
    
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
    
        Returns:
        If y is None, then it runs a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
    
        If y is not None, then it runs a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # Implementing the forward pass for the two-layer net, computing the       #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
      
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        hidden, cache_forw = affine_relu_forward(X, W1, b1) # output of hidden layer   
        
        scores,_ = affine_forward(hidden, W2, b2)  # Nx C
        
       

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implementing the backward pass for the two-layer net. Storing the loss   #
        # in the loss variable and gradients in the grads dictionary. Computed data #
        # loss using softmax, and grads[k] holds the gradients for self.params[k]  # .                   
        ############################################################################
       
        
        loss, dy = softmax_loss(scores, y)  # dy is the loss w.r.t input to softmax function
        loss += (self.reg/2) * ((W1**2).sum() + (W2**2).sum())  # adding regularisation loss
    
        cache2 = (hidden, W2, b2)
        dh, dW2, db2 = affine_backward(dy, cache2)
        dW2 += self.reg * W2
          
        dx, dW1, db1 = affine_relu_backward(dh, cache_forw)
        dW1 += self.reg * W1

        grads.update({'W1': dW1, 'b1':db1, 'W2': dW2, 'b2': db2})   # even updates the existing values

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. For a network with L layers,
    the architecture will be
    
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    The {...} block is repeated L - 1 times.
    
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32):
        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)  # input layer not considered
        self.dtype = dtype
        self.params = {}
        self.hidden_dims = hidden_dims

        ############################################################################
        # Initializing the weights and biases of the two-layer net. Weights        #
        # are initialized from a Gaussian with standard deviation equal to         #
        # weight_scale, and biases are initialized to zero. All weights and        #
        # biases are stored in the dictionary self.params, with first layer        #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2' etc                              #
        ############################################################################
   
        
        #input layer
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
        
        #hidden layers(except the last)
        for i in range(len(hidden_dims)-1):
            self.params['W'+str(i+2)] = np.random.normal(0.0, weight_scale, (hidden_dims[i], hidden_dims[i+1]))
            self.params['b'+str(i+2)] = np.zeros(hidden_dims[i+1])
        
        #last hidden_layer-output_layer
        self.params['W' + str(self.num_layers)] = np.random.normal(0.0, weight_scale, (hidden_dims[len(hidden_dims)-1], num_classes)) 
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
    
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        scores = None
        
        
           
        ############################################################################
        # Implementing the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        ############################################################################
       
        hidden_dims = self.hidden_dims  # I will be using it a lot in indexing
             
        cache_forws = []   # it will contain all the tuples. Will be needed for back propgaration
        hidden_out = []   # it will contain the output of hidden layers
            

        #Input to 1st hidden layer output
        hidden, cache_forw = affine_relu_forward(X, self.params['W1'], self.params['b1']) # output of hidden layer   
        cache_forws.append(cache_forw)
        hidden_out.append(hidden)
 
        #from 1st hidden layer output to last hidden layer output
        for i in range(len(hidden_dims)-1):
            hidden, cache_forw = affine_relu_forward(hidden_out[i], self.params['W'+str(i+2)], self.params['b'+str(i+2)]) 
            cache_forws.append(cache_forw)
            hidden_out.append(hidden)
            
    
        #from last hidden layer output to softmax input
        scores,_ = affine_forward(hidden_out[len(hidden_dims)-1], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])  # Nx C
             
        
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        
        
 
        
        ############################################################################
        # Implementing the backward pass for the fully-connected net. Storing the  #
        # loss in the loss variable and gradients in the grads dictionary. Computing #
        # data loss using softmax, and grads[k] holds gradients                    #
        ############################################################################
         
        doutputs = []   # it will store the derivatives of Loss wrt to output of each layer.
              
        loss, dout = softmax_loss(scores, y)  # dout is the loss w.r.t input to softmax function, y are the labels
        for i in range(self.num_layers):  # adding regularisation loss
            loss += (self.reg/2) * (self.params['W'+str(i+1)]**2).sum()
        doutputs.append(dout)    #  derivative wrt to input to softmax (output of output layer) is appended
            
          
        # update parameters between output layer and the last hidden layer
        cache = (hidden_out[len(hidden_dims)-1], self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
        dout,dW, db = affine_backward(doutputs[0], cache)
        doutputs.append(dout)  # adding the derivative wrt to output of last hidden layer
        dW += self.reg * self.params['W'+str(self.num_layers)]  # regression loss update
        grads.update({'W'+str(self.num_layers): dW, 'b'+str(self.num_layers): db})
     
        # update all the parameters between input layer and the last hidden layer 
        for i in range(self.num_layers -1):
            dout,dW, db = affine_relu_backward(doutputs[i+1], cache_forws[len(cache_forws)-(i+1)])         
            doutputs.append(dout)
            dW += self.reg * self.params['W'+str(self.num_layers - (i+1))]  # regression loss update
            grads.update({'W'+str(self.num_layers - (i+1)): dW, 'b'+str(self.num_layers - (i+1)): db})
            

        return loss, grads
