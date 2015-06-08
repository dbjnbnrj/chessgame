from train_helpers import *
from classifiers.linear_classifier import * 
from classifiers.k_nearest_neighbor import * 
import numpy

numToPiece = {0: 'Piece', 1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K'}


Xdata = ["../data/X_train_500.pkl", "../data/p1_X_500.pkl", "../data/p2_X_500.pkl", "../data/p3_X_500.pkl","../data/p4_X_500.pkl","../data/p5_X_500.pkl", "../data/p6_X_500.pkl"]
Ydata = ["../data/y_train_500.pkl", "../data/p1_y_500.pkl", "../data/p2_y_500.pkl", "../data/p3_y_500.pkl","../data/p4_y_500.pkl","../data/p5_y_500.pkl", "../data/p6_y_500.pkl"]

for net in [0]:
	X = get_data(Xdata[net])
	y = get_data(Ydata[net])
	totalsize = len(X) 
	
	trainsize= int ( 0.8 * totalsize)
	testsize = int ( 0.2 * trainsize)
	
	print Xdata[net]
	print totalsize, trainsize, testsize


	
	N, C, H, W = X.shape
	X = numpy.reshape(X, (N, C*H*W))

	X_train = X[:trainsize]
	y_train = y[:trainsize]
	X_val = X[trainsize : trainsize+testsize]
	y_val = y[trainsize : trainsize+testsize]

	classifiers = [KNearestNeighbor(), LinearSVM(), Softmax()]
	training = [X_train,X_train.transpose(), X_train.transpose() ]
	validation = [X_val, X_val.transpose(), X_val.transpose()]

	for idx, classifier in enumerate(classifiers):
		print "@ ",idx
		X_train = training[idx]
		X_val = validation[idx]

		print "---------"

		# Training
		print "Training...."
		loss = classifier.train(X_train, y_train)

		# Testing
		print "Predicting values ...."
		y_pred = classifier.predict(X_val)

		# Error rate

		print "Error rate..."
		classifier.error(y_val, y_pred)

		print "---------"

