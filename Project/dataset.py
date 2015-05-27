import chess, chess.pgn
import numpy as np
import pickle
import os
import math
from helper import *

#NUM_TRAIN = 0.8* NUM_GAMES
#NUM_VALID = 0.2* NUM_GAMES
#NUM_TEST = NUM_GAMES - (NUM_TRAIN + NUM_VALID)

# Move-out
X_train, y_train = [], []

# Move-in
p1_X, p2_X, p3_X = [], [], []
p4_X, p5_X, p6_X = [], [], []
p1_y, p2_y, p3_y = [], [], []
p4_y, p5_y, p6_y = [], [], []

gameidx = 1

for g in read_games("data/finaldata.pgn"):
	print "@game:", gameidx
	gameidx+=1
	node = g
	moveidx = 0
	while node.variations:
		b = node.board()
		node = node.variation(0)
		from_square = node.move.from_square
		to_square = node.move.to_square

		if moveidx % 2 == 0:
			im = convert_bitboard_to_image(b)	
		else:
			im = flip_image( convert_bitboard_to_image(b) )
			im = flip_color(im)
			from_square = 64 - from_square
			to_square = 64 - to_square

		im = np.rollaxis(im, 2, 0)

		X_train.append(im)
		y_train.append(from_square)

		piece = b.piece_type_at(chess.SQUARES[node.move.from_square])
		
		p_X = "p%d_X" % (piece)
		p_X = eval(p_X)
		p_X.append(im)
		
		p_y = "p%d_y" % (piece )
		p_y = eval(p_y)
		p_y.append(to_square)

		moveidx +=1

# Move-out
X_train, y_train = np.array(X_train), np.array(y_train)


# Move-in
p1_X, p2_X, p3_X = np.array(p1_X), np.array(p2_X), np.array(p3_X)
p4_X, p5_X, p6_X = np.array(p4_X), np.array(p5_X), np.array(p6_X)
p1_y, p2_y, p3_y = np.array(p1_y), np.array(p2_y), np.array(p3_y)
p4_y, p5_y, p6_y = np.array(p4_y), np.array(p5_y), np.array(p6_y)

print "Processed %d games out of %d" % (NUM_GAMES, NUM_GAMES)
print "Saving data..."

print "Saving X_train array..."
output = open('X_train_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(X_train, output)
output.close()

print "Saving y_train array..."
output = open('y_train_%d.pkl' % NUM_GAMES, 'wb')
pickle.dump(y_train, output)
output.close()

for i in xrange(6):
	output_array = "p%d_X" % (i + 1)
	print "Saving %s array..." % output_array
	output_array = eval(output_array)
	output = open('p%d_X_%d.pkl' % (i + 1, NUM_GAMES), 'wb') 
	pickle.dump(output_array, output)
	output.close()

	output_array = "p%d_y" % (i + 1)
	print "Saving %s array..." % output_array
	output_array = eval(output_array)
	output = open('p%d_y_%d.pkl' % (i + 1, NUM_GAMES), 'wb') 
	pickle.dump(output_array, output)
	output.close()

print "Done!"
