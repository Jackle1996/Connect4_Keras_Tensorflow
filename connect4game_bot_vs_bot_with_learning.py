# source: https://github.com/KeithGalli/Connect4-Python/blob/master/connect4.py
import numpy as np
import pygame
import sys
import math

import keras

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

def print_board(board):
	print(np.flip(board, 0))

def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

def draw_board(board):
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):		
			if board[r][c] == 1:
				pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
			elif board[r][c] == 2: 
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
	pygame.display.update()

def determine_move_deeplearningbot(board, model): # bot move
	#print("-------- calculating win chances----------")
	predictedChances = np.zeros(7)
	for column in range(0, 7):
	
		# predict value
		inputfeatures = np.copy(board)

		if is_valid_location(inputfeatures, column):
			row = get_next_open_row(inputfeatures, column)
			inputfeatures[row, column] = 2

			inputfeatures[inputfeatures == 2] = -1
			inputfeatures = np.flip(inputfeatures, 0) # board is reversed!
			inputfeatures = inputfeatures.reshape(1,42) # flatten to correct input vector
			predictions = model.predict(inputfeatures)
			#print("prediction for column ",column, "is ", predictions)
			predictedChances[column] = predictions[0, 2]
	
	
	
	col = np.argmax(predictedChances)
	print("best choice is column ", col)
	return col


board = create_board()
print(sys.argv[0]) # file.py

model1 = keras.models.load_model(sys.argv[1])
model2 = keras.models.load_model(sys.argv[2])
	
print_board(board)
game_over = False
turn = 0
winner = 0


pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)


while not game_over:
	pygame.display.update()
	pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))

	# Ask for Player 1 Input #####################################################################
	# ai plays	
	if not game_over:
		col = determine_move_deeplearningbot(board, model1)
		
		if is_valid_location(board, col) and not game_over:
			row = get_next_open_row(board, col)
			drop_piece(board, row, col, 1)

			if winning_move(board, 1):
				label = myfont.render(sys.argv[1]+" wins !!", 1, RED)
				screen.blit(label, (40,10))
				game_over = True
				winner = 1
	
	draw_board(board)
	pygame.time.wait(30)

	# Ask for Player 2 Input #####################################################################
	# ai plays	
	if not game_over:
		col = determine_move_deeplearningbot(board, model2)
		
		if is_valid_location(board, col) and not game_over:
			row = get_next_open_row(board, col)
			drop_piece(board, row, col, 2)

			if winning_move(board, 2):
				label = myfont.render(sys.argv[2]+" wins !!", 1, YELLOW)
				screen.blit(label, (40,10))
				game_over = True
				winner = -1
	
	draw_board(board)
	pygame.time.wait(30)

	if game_over:
		print("saving winning board")
		# convert into correct feature format
		board[board == 2] = -1
		board = np.flip(board, 0) # board is reversed!
		board = board.reshape(1,42) # flatten to correct input vector
		board = np.append(board, winner) # add winner -> reshapes board for some reason
		# save board
		np.savetxt("endboard.csv", [board], fmt='%1.1f', delimiter=",")
		pygame.time.wait(100)