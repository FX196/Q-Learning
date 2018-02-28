import random
import copy

EMPTY_CELL = "    "


class Game(object):
    def __init__(self, size):
        self.size = size
        self.board = get_board(size)
        self.score = 0
        self.game_over = False
        self.new_cell()

    def display(self):
        print("Score: %d" % self.score)
        print("-"*21)
        for row in range(self.size):
            print("|", end="")
            for col in range(self.size):
                print("{0:{width}}".format(self.board[row][col], width=4), end="|")
            print()
            print("-" * 21)

    # add new cell with value of 2 or 4 to a random empty cell on the board
    # returns: True if adding successful, False if unsuccessful i.e. no empty cells
    def new_cell(self):
        board = self.board
        empty_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == EMPTY_CELL:
                    empty_cells.append((i, j))
        if empty_cells:
            r, c = empty_cells[random.randint(0, len(empty_cells) - 1)]
            board[r][c] = random.randint(1, 2) * 2
            return True
        else:
            return False

    # process action, 0 for up, 1 for down, 2 for left, 3 for right
    # returns: True if move legal, false if otherwize
    def step(self, action):
        temp_board = []
        if action == "0":
            for col in transpose(self.board):
                temp_board.append(reduce(col))
                temp_board = transpose(temp_board)
        elif action == "1":
            for col in transpose(self.board):
                temp_board.append(reduce(col[::-1])[::-1])
                temp_board = transpose(temp_board)
        elif action == "2":
            for row in self.board:
                temp_board.append(reduce(row))
        elif action == "3":
            for row in self.board:
                temp_board.append(reduce(row[::-1])[::-1])
        else:
            print("Action invalid")
        if temp_board != self.board:
            self.board = temp_board
            self.new_cell()
            self.display()


# returns a board full of empty cells with side length equal to size
# returns: 2D list representing the board
def get_board(size):
    board = []
    for i in range(size):
        board.append([])
        for j in range(size):
            board[i].append(EMPTY_CELL)
    return board


# reduces row i.e. move and merge cells to the left
# returns: list representing reduced row
def reduce(row):
    ind = 0
    row_length = len(row)
    row = copy.deepcopy(row)
    while ind < len(row)-1:
        if row[ind] == EMPTY_CELL:
            row.pop(ind)
        elif row[ind+1] == EMPTY_CELL:
            row.pop(ind+1)
        elif row[ind] == row[ind+1]:
            row[ind] += row.pop(ind+1)
            ind += 1
        else:
            ind += 1
    while len(row) < row_length:
        row.append(EMPTY_CELL)
    return row


# transposes board to simplify step process
# returns: 2D list representing transposed board
def transpose(board):
    board = copy.deepcopy(board)
    for i in range(len(board)):
        for j in range(i):
            board[i][j], board[j][i] = board[j][i], board[i][j]
    return board


if __name__ == "__main__":
    data = Game(4)
    data.display()
    while not data.game_over:
        action = input()
        data.step(action)

