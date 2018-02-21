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
        board.append([])
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


def mouse_pressed(event, data):
    # use event.x and event.y
    pass


def key_pressed(event, data):
    # use event.char and event.keysym
    pass


def draw_board(canvas, data):
    pass


def show_score(canvas, data):
    pass


def redrawAll(canvas, data):
    draw_board(canvas, data)
    show_score(canvas, data)


####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mouse_pressedWrapper(event, canvas, data):
        mouse_pressed(event, data)
        redrawAllWrapper(canvas, data)

    def key_pressedWrapper(event, canvas, data):
        key_pressed(event, data)
        redrawAllWrapper(canvas, data)

    # Set up data and call init
    class Struct(object): pass

    data = Struct()
    data.width = width
    data.height = height
    root = Tk()
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mouse_pressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            key_pressedWrapper(event, canvas, data))
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(500, 600)