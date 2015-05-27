import numpy as n
import matplotlib.pyplot as plt
from time import time
import pickle
from layers import *
from fast_layers import *
from classifier_trainer import ClassifierTrainer
from gradient_check import eval_numerical_gradient_array


def get_data(path):
	f = open(path)
	return pickle.load(f)

def get_params(pfn):
	params = {'Piece' : pfn(), 'P': pfn(), 'R': pfn(), 'N': pfn(), 'Q':pfn(), 'K': pfn()}
	return params

def plot(loss_history, train_acc_history, val_acc_history):
	plt.subplot(2, 1, 1)
	plt.plot(train_acc_history)
	plt.plot(val_acc_history)
	plt.title('accuracy vs time')
	plt.legend(['train', 'val'], loc=4)
	plt.xlabel('epoch')
	plt.ylabel('classification accuracy')

	plt.subplot(2, 1, 2)
	plt.plot(loss_history)
	plt.title('loss vs time')
	plt.xlabel('iteration')
	plt.ylabel('loss')
	plt.show()

def train(trainX, trainY, valX,valY, params, convnet ):
	trainer = ClassifierTrainer()
	best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
		trainX, trainY, valX, valY, params, convnet, 
		reg = 0., learning_rate= 0.0001, batch_size=250, num_epochs=15, learning_rate_decay=0.99, update='rmsprop', verbose=True, dropout=1.0)
	return best_model, loss_history, train_acc_history, val_acc_history 