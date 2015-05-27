import chess, chess.pgn
import numpy as np
import pickle
import os

PIECE_TO_INDEX = {'P' : 0, 'R' : 1, 'N' : 2, 'B' : 3, 'Q' : 4, 'K' : 5}

''' Helpers for train '''

def get_dataset(path):
	f = open(path)
	return pickle.load(f)


''' Helpers for create_dataset '''

def display(move_selector, piece_selector):
	print "\n Move selector : "
	print move_selector[0][0][0], move_selector[0][1]

	print "\n Piece selector : "
	for k in piece_selector.keys():
		print piece_selector[k]

def maketuple(x, y , train, validate, test):
	ranges = train
	train_set = ( x[ranges[0]: ranges[1]], y[ranges[0]: ranges[1]]) 
	ranges = validate
	validate_set = ( x[ranges[0]: ranges[1]], y[ranges[0]: ranges[1]])
	ranges = test
	test_set = ( x[ranges[0]: ranges[1]], y[ranges[0]: ranges[1]]) 
	return (train_set, validate_set, test_set)


def flip_image(im):
	return im[::-1, :, :]

def flip_color(im):
	indices_white = np.where(im == 1)
	indices_black = np.where(im == -1)
	im[indices_white] = -1
	im[indices_black] = 1
	return im

def read_games(fn):
   f = open(fn)
   while True:
	    try:
	        g = chess.pgn.read_game(f)
	    except KeyboardInterrupt:
	        raise
	    except:
	        continue

	    if not g:
	        break
	    
	    yield g

def convert_bitboard_to_image(board):
	BOARD_SIZE = (8,8, 6)
	im2d = np.array(list(str(board).replace('\n', '').replace(' ', ''))).reshape((8, 8))
	im = np.zeros(BOARD_SIZE)

	for i in xrange(BOARD_SIZE[0]):
		for j in xrange(BOARD_SIZE[1]):
			piece = im2d[i, j]
			if piece == '.': continue
			if piece.isupper():
				im[i, j, PIECE_TO_INDEX[piece.upper()]] = 1
			else:
				im[i, j, PIECE_TO_INDEX[piece.upper()]] = -1

	return im
