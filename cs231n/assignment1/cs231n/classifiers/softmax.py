from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.reshape(np.amax(scores, axis = 1), [num_train, 1]) #numerical reason
    scores_expsum = np.sum(np.exp(scores), axis = 1)
    for i in xrange(num_train):

 
      cor_ex = np.exp(scores[i, y[i]])
      loss += - np.log( cor_ex / scores_expsum[i])

      # grad
      # for correct class
      dW[:, y[i]] += (-1) * (scores_expsum[i] - cor_ex) / scores_expsum[i] * X[i]
      for j in xrange(num_classes):
          # pass correct class gradient
          if j == y[i]:
              continue
          # for incorrect classes
          dW[:, j] += np.exp(scores[i,j]) / scores_expsum[i] * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W            
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.reshape(np.amax(scores, axis = 1), [num_train, 1]) #numerical reason
    scores = np.exp(scores)
    scores_expsum = np.sum(scores, axis = 1)
    selection_matrix = [range(num_train),y] 
    score_cor = scores[selection_matrix]   
    loss = -np.log(score_cor/scores_expsum)
    loss = np.sum(loss)/num_train + reg*np.sum(W*W)

    s = scores/scores_expsum.reshape(num_train, 1)
    s[selection_matrix] = scores[selection_matrix]/scores_expsum-1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = X.T.dot(s)
    dW /= num_train

    dW += 2*reg*W
    return loss, dW
