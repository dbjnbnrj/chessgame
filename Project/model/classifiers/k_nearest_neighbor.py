import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)
    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i, j] = np.sqrt(np.sum((X[i, :] - self.X_train[j, :]) ** 2))
    return dists

  def compute_distances_one_loop(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      dists[i, :] = np.sqrt(np.sum((self.X_train - X[i, :]) ** 2, axis=1))
    return dists

  def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    X_norms = np.sum(X ** 2, axis=1, keepdims=True)
    X_train_norms = np.sum(self.X_train ** 2, axis=1)
    cross = -2.0 * X.dot(self.X_train.T)
    dists = np.sqrt(X_norms + cross + X_train_norms)
    return dists

  def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      closest_y = []
      sortix = np.argsort(dists[i, :])
      closest_y = self.y_train[sortix[:min(k, len(sortix))]]
      counts = {}
      for y in closest_y:
        counts[y] = counts.get(y, 0) + 1
      tuples = sorted([(val, key) for key, val in counts.iteritems()], reverse=True)
      y_pred[i] = tuples[0][1]
    return y_pred
  
  def error(self, y, y_pred):
    error = 0
    total = y.shape[0]
    for i in range(0, total):
      if (y[i] !=y_pred[i]):
        error+=1
    print "Correctly classified ", error , "out of ", total, "examples "
    return


