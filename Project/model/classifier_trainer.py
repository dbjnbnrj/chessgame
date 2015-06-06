import numpy as np
from util import *

class ClassifierTrainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

  def train(self, X, y, X_val, y_val, 
            model, loss_function, 
            reg=0.0, dropout=1.0,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,
            update='momentum', sample_batches=True,
            num_epochs=30, batch_size=100, acc_frequency=None,
            augment_fn=None, predict_fn=None,
            verbose=False):

    N = X.shape[0]

    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_acc = 0.0
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    for it in xrange(num_iters):
      if it % 10 == 0:  print 'starting iteration ', it

      # get batch of data
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch = X[batch_mask]
        y_batch = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch = X
        y_batch = y

      # Maybe perform data augmentation
      if augment_fn is not None:
        X_batch = augment_fn(X_batch)

      # evaluate cost and gradient
      # cost, grads = loss_function(X_batch, model, y_batch, reg=reg, dropout=dropout)
      cost, grads = loss_function(X_batch, model, y_batch, reg=reg)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        elif update == 'momentum':
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = np.zeros_like(grads[p]) # you can remove this after
          dx = momentum * self.step_cache[p] - learning_rate * grads[p]
          self.step_cache[p] = dx
        elif update == 'rmsprop':
          decay_rate = 0.99 # you could also make this an option
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
          dx = np.zeros_like(grads[p]) # you can remove this after
          self.step_cache[p] = self.step_cache[p] * decay_rate + (1.0 - decay_rate) * grads[p] ** 2
          dx = -(learning_rate * grads[p]) / np.sqrt(self.step_cache[p] + 1e-8)
        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        model[p] += dx

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate
          learning_rate *= learning_rate_decay
          epoch += 1

        # evaluate train accuracy
        if N > 1000:
          train_mask = np.random.choice(N, 1000)
          X_train_subset = X[train_mask]
          y_train_subset = y[train_mask]
        else:
          X_train_subset = X
          y_train_subset = y
        # Computing a forward pass with a batch size of 1000 will is no good,
        # so we batch it
        y_pred_train = []
        for i in xrange(X_train_subset.shape[0] / 100):
          X_train_slice = X_train_subset[i*100:(i+1)*100]
          if predict_fn is not None:
            X_train_slice = predict_fn(X_train_slice)
          scores = loss_function(X_train_slice, model)
          y_pred_train.append(np.argmax(scores, axis=1))
        y_pred_train = np.hstack(y_pred_train)
        train_acc = np.mean(y_pred_train == y_train_subset)
        train_acc_history.append(train_acc)

        # evaluate val accuracy, but split the validation set into batches
        y_pred_val = []
        for i in xrange(X_val.shape[0] / 100):
          X_val_slice = X_val[i*100:(i+1)*100]
          if predict_fn is not None:
            X_val_slice = predict_fn(X_val_slice)
          scores = loss_function(X_val_slice, model)
          y_pred_val.append(np.argmax(scores, axis=1))

        y_pred_val = np.hstack(y_pred_val)
        numI = X_val.shape[0] / 100
        #print y_pred_val.shape, y_val[:numI * 100].shape
        val_acc = np.mean(y_pred_val ==  y_val[:numI * 100])
        #print val_acc
        val_acc_history.append(val_acc)
        
        # keep track of the best model based on validation accuracy
        if val_acc > best_val_acc:
          # make a copy of the model
          best_val_acc = val_acc
          best_model = {}
          for p in model:
            best_model[p] = model[p].copy()

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))

    if verbose:
      print 'finished optimization. best validation accuracy: %f' % (best_val_acc, )
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history
