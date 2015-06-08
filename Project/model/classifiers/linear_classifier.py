import numpy as np

class LinearClassifier:

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    dim, num_train = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    
    print "num_classes", num_classes
    if self.W is None:
      self.W = np.random.randn(num_classes, dim) * 0.001


    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      batch_mask = np.random.choice(num_train, batch_size)
      X_batch = X[:,batch_mask]
      y_batch = y[batch_mask]

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      step = -learning_rate * grad
      self.W += step

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[0])
    print "y_pred.shape", y_pred.shape
    scores = self.W.dot(X)
    print "scores.shapes ", scores.shape
    y_pred = np.argmax(scores, axis=0) # top scoring class
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    pass


  def error(self, y, y_pred):
    error = 0
    total = y.shape[0]
    total2 = y_pred.shape[0]
    print total , total2
    for i in range(0, total):
      if (y[i] !=y_pred[i]):
        error+=1
    print "Correctly classified",  total-error, " of the ", total, "examples "
    return


class LinearSVM(LinearClassifier):
  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


def svm_loss_vectorized(W, X, y, reg):

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  D, num_train = X.shape
  scores = W.dot(X)
  correct_class_scores = scores[y, range(num_train)]
  margins = np.maximum(0, scores - correct_class_scores + 1.0)
  margins[y, range(num_train)] = 0

  loss_cost = np.sum(margins) / num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost + loss_reg
  num_pos = np.sum(margins > 0, axis=0) # number of positive losses

  dscores = np.zeros(scores.shape)
  dscores[margins > 0] = 1
  dscores[y, range(num_train)] = -num_pos

  dW = dscores.dot(X.T) / num_train + reg * W

  return loss, dW


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_loss_vectorized(W, X, y, reg):
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  D, num_train = X.shape
  scores = W.dot(X) # C x N

  scores -= np.max(scores, axis = 0)
  p = np.exp(scores)
  p /= np.sum(p, axis = 0)

  loss_cost = -np.sum(np.log(p[y, range(y.size)])) / num_train
  loss_reg = 0.5 * reg * np.sum(W * W)
  loss = loss_cost + loss_reg

  dscores = p
  dscores[y, range(y.size)] -= 1.0
  dW = dscores.dot(X.T) / num_train + reg * W

  return loss, dW
