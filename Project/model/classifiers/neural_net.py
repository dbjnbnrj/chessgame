import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  
  i1 = X.dot(W1) + b1
  o1 = np.where(i1> 0, i1, 0) # Otherwise causing problems with the shape
  scores = o1.dot(W2) + b2

  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  f = scores.T - np.max(scores, axis =1) # Regularizing
  f = np.exp(f)
  p = f / np.sum(f, axis=0)

  # loss function
  distr = np.arange(N) # random distribution
  loss = np.mean(-np.log(p[y, distr]))
  loss += (0.5*reg) * np.sum(W1 * W1)
  loss += (0.5*reg) * np.sum(W2 * W2)

  # compute the gradients
  grads = {}

  df = p  # (C, N)
  df[y, distr] -= 1
  # (H, C) = ((C, N) x (N, H)).T
  dW2 = df.dot(o1).T / N  # (H, C)
  dW2 += reg * W2
  grads['W2'] = dW2

  # C = (C, N)
  db2 = np.mean(df, axis=1)  # C
  grads['b2'] = db2

  # (N, H) =  (H, C)
  dresp1 = W2.dot(df).T / N
  ds1 = np.where(i1 > 0, dresp1, 0)  # (N, H)
  dW1 = X.T.dot(ds1)  # (D, H)
  dW1 += reg * W1
  grads['W1'] = dW1

  db1 = np.sum(ds1, axis=0)  # H
  grads['b1'] = db1
  return loss, grads