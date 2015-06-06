import numpy as np
import chess
import pickle
import random
import time
import traceback
import re
import string
import math
from helper import *
from chess import pgn
from layers import *
from fast_layers import *
from classifiers.convnet import *
from classifier_trainer import ClassifierTrainer
from time import time

trained_models = {}

class Player(object):
	def move(self, gamenode):
		pass

class Human(Player):
	def move(self, game_node):
		game_node_curr = game_node
		bb = game_node.board()

		def get_move(move_str):
			try:
				move = chess.Move.from_uci(move_str)
			except:
				print 'wrong move'
				return False
			if move not in bb.legal_moves:
				print 'not a legal move'
				return False
			else:
				return move
			while True:
				print 'your turn:'
				move = get_move(raw_input())
				if move:
					break

			game_new = chess.pgn.GameNode()
			game_new.parent = game_node_curr
			print move
			game_new.move = move
			
			return game_new

class Computer(Player):
	def move(self, gn_current):
		bb = gn_current.board()
		print bb

		im = convert_bitboard_to_image(bb)
		im = np.rollaxis(im, 2, 0)

		move_str = self.predictMove(im)
		move = chess.Move.from_uci(move_str)

		if move not in bb.legal_moves:
		    print "NOT A LEGAL MOVE"

		gn_new = chess.pgn.GameNode()
		gn_new.parent = gn_current
		gn_new.move = move
		return gn_new

	def predictMove(self, img):
	    modelScores = {}
	    scores = three_layer_convnet(np.array([img]), trained_models['Piece'])


	    for key in trained_models.keys():
	        if key != 'Piece':
	            modelScores[key] = three_layer_convnet(np.array([img]), trained_models[key])

	    availablePiecesBoard = clip_pieces_single(scores, img) # (1, 64) size
	    
	    #print availablePiecesBoard

	    maxScore = 0
	    maxFromCoordinate, maxToCoordinate = -1, -1
	    availablePiecesBoard = np.reshape(availablePiecesBoard, (64))
	    for i in xrange(64):
	        coordinate = scoreToCoordinateIndex(i)
	        if availablePiecesBoard[i] != 0:
	            pieceType = INDEX_TO_PIECE[np.argmax(img[:, coordinate[0], coordinate[1]])]
	            availableMovesBoard = clip_moves(modelScores[pieceType], img, coordinate)
	            #print "Piece Type: ", pieceType, availableMovesBoard
	            composedScore = np.max(boardToScores(availableMovesBoard)) * availablePiecesBoard[i]
	            if composedScore > maxScore:
	                maxScore = composedScore
	                maxFromCoordinate, maxToCoordinate = coordinate, scoreToCoordinateIndex(np.argmax(boardToScores(availableMovesBoard)))

	    maxFromCoordinate = coord2d_to_chess_coord(maxFromCoordinate)
	    maxToCoordinate = coord2d_to_chess_coord(maxToCoordinate)   
	    return maxFromCoordinate + maxToCoordinate


def game(player1, player2):
	gn_current = chess.pgn.Game()
	maxn = 10 ** (2.0 + random.random() * 1.0 ) # max nodes for sunfish

	times = {'A': 0.0 , 'B': 0.0}
	while True:
		for side, player in [('A', player1), ('B', player2)]:
			try:
				gn_current = player.move(gn_current)
			except KeyboardInterrupt:
				return
			except:
				traceback.print_exc()
				return side + '-exception'

			print '----- Player %s: %s ' % (side,  gn_current)
			

			b = str(gn_current.board())
			print b

			print '------ End Turn --------'

			if gn_current.board().is_checkmate():
				return side
			elif gn_current.board().is_stalemate():
				return '-'
			elif gn_current.board().can_claim_fifty_moves():
				return '-'
			elif b.find('K') == -1 or b.find('k') == -1:
				return side

def setup():
	model_names = ["piece", "pawn", "rook","knight", "bishop", "queen", "king"]
	names = ['Piece', 'P', 'R', 'N', 'B', 'Q', 'K']
	for idx, name in enumerate(model_names):
		path = "../data/model_%s.pkl" % name
		f = open(path)
		trained_models[names[idx]] = pickle.load(f)

if __name__ == '__main__':
	setup()
	game( Computer(), Human() ) 
	