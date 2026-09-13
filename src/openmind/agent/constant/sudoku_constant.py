NAME = "sudoku"
SEPARATOR = "/"
BOX = 3
SIZE = BOX * BOX
CELLS = SIZE * SIZE
DIGITS = tuple(range(1, SIZE + 1))
PUZZLE = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
EMPTY_MARK = "."
CLUE_MARKS = tuple(str(digit) for digit in DIGITS)
EMPTY = None
UNSET = None

COLLECTION_SUFFIX = ".txt"
EULER_HEADER = "Grid"
EULER_EMPTY_MARK = "0"

SOLVED = 1.0
CERTAIN = 1.0

CELL = "cell"
TURN = "turn"
PAYOFF = "payoff"
PLAYER = "solver"
FILL = "fill"
