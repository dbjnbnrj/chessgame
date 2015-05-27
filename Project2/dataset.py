import chess, chess.pgn
import numpy as np
import pickle
import os
import math
import cPickle
from helper import *

num_games = 100
train_range = (0,  int ( 0.7 * num_games ))
valid_range = ( int ( 0.7 * num_games ), int ( 0.8 * num_games ) )
test_range = ( int ( 0.8 * num_games ), num_games-1 )

num_test = int ( 0.7 * num_games ) 
data_file = "data/dataset_"+str(num_games)+".pgn"

piece_selector_x = []
piece_selector_y = []
move_selector_x = []
move_selector_y = []

for g in read_games(data_file):
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
			im = flip_color(flip_image( convert_bitboard_to_image(b) ))
			from_square, to_square = 64 - from_square, 64 - to_square

		im = np.rollaxis(im, 2, 0)
		piece_selector_x.append(im)
		piece_selector_y.append(from_square)
		move_selector_x.append(im)
		move_selector_y.append(to_square)
		moveidx +=1



move_selector = []
move_selector.append((move_selector_x, move_selector_y))

print "Saving piece selector network ..."
train_set = ( piece_selector_x[ train_range[0]:train_range[1] ],  piece_selector_y[ train_range[0]:train_range[1] ] )
valid_set = ( piece_selector_x[ valid_range[0]:valid_range[1] ], piece_selector_y[ valid_range[0]:valid_range[1] ] )
test_set =  ( piece_selector_x[ test_range[0]:test_range[1] ], piece_selector_y[ test_range[0]:test_range[1] ] )

final_piece_selector = (train_set, valid_set, test_set)

output = open('piece_selector.pkl', 'wb')
pickle.dump(final_piece_selector, output)
output.close()

print "Saving move selector network ..."

train_set = ( move_selector_x[ train_range[0]:train_range[1] ],  move_selector_y[ train_range[0]:train_range[1] ] )
valid_set = ( move_selector_x[ valid_range[0]:valid_range[1] ], move_selector_y[ valid_range[0]:valid_range[1] ] )
test_set =  ( move_selector_x[ test_range[0]:test_range[1] ], move_selector_y[ test_range[0]:test_range[1] ] )

final_move_selector = (train_set, valid_set, test_set)

output = open('move_selector.pkl', 'wb')
pickle.dump(final_move_selector, output)
output.close()



