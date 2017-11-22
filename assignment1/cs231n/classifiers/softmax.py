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
  num_classes = W.shape[1] #it is 10
  num_train = X.shape[0] # it is 500
  for i in range(num_train):
  	scores = X[i].dot(W)  #scores is (1 by 10)
  	individual_prob = np.exp(scores)/sum(np.exp(scores))
  	loss += -np.log(individual_prob[y[i]])
  	for j in range(num_classes):
  		if j==y[i]:
  			dW[:,j] += (individual_prob[j]-1)*X[i]
  		else:
  			dW[:,j] += individual_prob[j]*X[i]
  


  loss /= num_train
  dW /= num_train
  dW += 2*reg*W
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  temp1 = np.exp(scores)

  individual_prob = temp1/temp1.sum(axis=1)[:,None]
  loss += np.sum(-np.log(individual_prob[np.arange(num_train),y]))

  I = np.zeros_like(individual_prob)
  I[np.arange(num_train), y] = 1
  dW = X.T.dot(individual_prob-I)

  loss /= num_train
  dW /= num_train
  dW += 2*reg*W
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

