import random

EMPTY_CELL = "    "


class Game(object):
    def __init__(self, size):
        self.size = size
        self.board = get_board(size)
        self.score = 0
        self.game_over = False


    def display(self):
        print("Score: %d" % self.score)
        print("-"*21)
        for row in range(self.size):
            print("|", end="")
            for col in range(self.size):
                print("{0:{width}}".format(self.board[row][col], width=4), end="|")
            print()
            print("-" * 21)

    def step(self, action):
        moved = True
        if action == "0":
            pass
        elif action == "1":
            pass
        elif action == "2":
            pass
        elif action == "3":
            pass
        else:
            print("Action invalid")
        if moved:
            new_cell(self)
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


# add new cell with value of 2 or 4 to a random empty cell on the board
# returns: True if adding successful, False if unsuccessful i.e. no empty cells
def new_cell(data):
    board = data.board
    empty_cells = []
    for i in range(data.size):
        for j in range(data.size):
            if board[i][j] == EMPTY_CELL:
                empty_cells.append((i,j))
    if empty_cells:
        r, c = empty_cells[random.randint(0,len(empty_cells)-1)]
        board[r][c] = random.randint(1, 2) * 2
        return True
    else:
        return False


# process action
# returns: True if action legal, False if not legal



if __name__ == "__main__":
    data = Game(4)
    data.display()
    while not data.game_over:
        action = input()
        data.step(action)