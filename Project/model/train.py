from train_helpers import *
from classifiers.neural_net import * 
from classifiers.convnet import *
import numpy

def two_layer_affine_net():
	# Only For Neural Networks
	N, C, H, W = X_train.shape
	X_train = numpy.reshape(X_train, (N, C*H*W))

	N, C, H, W = X_val.shape
	X_val = numpy.reshape(X_val, (N, C*H*W))

	inp ,hidden,out =  (384, 800, 64)
	print "Affine Neural Network Layout --- ", inp ,hidden,out
	trainer = ClassifierTrainer()
	result = trainer.train(X_train, y_train, X_val, y_val, init_two_layer_model(inp ,hidden,out), two_layer_net,
		reg=0.,learning_rate=0.001, learning_rate_decay=0.01, update='sgd', sample_batches=False, 
		num_epochs=10,verbose=False)
	return result

def two_layer_conv_net():
	print "Conv-Relu + Affine Neural Network Layout --- "
	result = train(X_train, y_train, X_val, y_val, init_two_layer_convnet(), two_layer_convnet)
	return result

def three_layer_conv_net():
	print " [conv - relu] - [affine - relu] - [affine - softmax]"
	result = train(X_train, y_train, X_val, y_val, init_three_layer_convnet(), three_layer_convnet)
	return result

def five_layer_conv_net():
	print "[conv - relu - pool] x 3 - [affine - relu - dropout] - affine - softmax"
	result = train(X_train, y_train, X_val, y_val, init_five_layer_convnet(), five_layer_convnet)
	return result

def chess_net():
	print "[conv - relu - pool] x 3 - [affine - relu - dropout] - affine - softmax"
	result = train(X_train, y_train, X_val, y_val, init_chess_convnet(), chess_convnet)
	return result

Xdata = ["../data/X_train_500.pkl", "../data/p1_X_500.pkl", "../data/p2_X_500.pkl", "../data/p3_X_500.pkl","../data/p4_X_500.pkl","../data/p5_X_500.pkl", "../data/p6_X_500.pkl"]
Ydata = ["../data/y_train_500.pkl", "../data/p1_y_500.pkl", "../data/p2_y_500.pkl", "../data/p3_y_500.pkl","../data/p4_y_500.pkl","../data/p5_y_500.pkl", "../data/p6_y_500.pkl"]

for net in [0]:
	X = get_data(Xdata[net])
	y = get_data(Ydata[net])
	
	total = len(X)
	trainsize= int ( 0.9 * total ) 
	testsize = int ( 0.1* total )

	print "Total: ", total, "Training size: ",  trainsize, "Testing size: ",  testsize

	X_train = X[:trainsize]
	y_train = y[:trainsize]
	X_val = X[trainsize:trainsize+testsize]
	y_val = y[trainsize:trainsize+testsize]

	best_model, loss_history, train_acc_history, val_acc_history = chess_net()
	
	#newfname = Xdata[net].replace("X", "model")
	#print "Saving result in ...", newfname
	#save_pickled(best_model, newfname)

	print 'Final loss: ' % (loss_history[-1])
	print 'Final validation acc: ', val_acc_history[-1]
	print 'Final train acc: ', train_acc_history[-1]

	plot(loss_history, train_acc_history, val_acc_history)



