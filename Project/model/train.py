from train_helpers import *
from classifiers.convnet import *

init_convnet = init_three_layer_convnet
convnet = three_layer_convnet

numToPiece = {0: 'Piece', 1: 'P', 2: 'R', 3: 'N', 4: 'B', 5: 'Q', 6: 'K'}

netsToTrain = [0]

data = ()
for net in netsToTrain:
	models = get_params(init_convnet)
	X = get_data("../data/X_train_8888.pkl")
	y = get_data("../data/y_train_8888.pkl")
	
	trainsize= 1000
	testsize = 500

	print len(X), len(y), trainsize, testsize

	X_train = X[:trainsize]
	y_train = y[:trainsize]
	X_val = X[trainsize:trainsize+testsize]
	y_val = y[trainsize:trainsize+testsize]
	# print X_train.shape
	# print X_val.shape
	# print y_val.shape
	# print numToPiece[net]

	results = train(X_train, y_train, X_val, y_val, models[numToPiece[net]], convnet)

	plot(results[1], results[2], results[3])



