from tkinter import *
from random import randint

####################################
# customize these functions
####################################


def init(data):
    # load data.xyz as appropriate
    data.game_size = 4
    data.board = get_board(data)
    data.top_height = 100
    data.margin = 15
    data.cell_width = 106.25


def get_board(data):
    board = []
    for row in range(data.game_size):
        board.append(["    "])
        for col in range(data.game_size):
            board[row].append([])
    return board


def new_tile(data):
    board = data.board
    empty_spots = []
    for row in range(data.game_size):
        for col in range(data.game_size):
            if not board[row][col]:
                empty_spots.append((row, col))
    r, c = empty_spots[randint(0, len(empty_spots)-1)]
    num = randint(1, 2) * 2
    board[r][c] = num


def play_action(data, action):
    """0: up 1: down 2: left 3: right"""