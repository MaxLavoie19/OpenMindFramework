DEFAULT_GAMES = 100
DEFAULT_ITERATIONS = 200
DEFAULT_POSITIONS = 100
DEFAULT_BUDGETS = (10, 20, 50, 100, 200, 500)
DEFAULT_SEED = 1
ALL_POSITIONS = "all"

RANDOM_OPPONENT = "random"
UNTRAINED_OPPONENT = "untrained MCTS"

#: With a reference search, actions whose value is within this of the best are optimal.
REFERENCE_TOLERANCE = 0.05
#: Random games tried per position wanted, at most, when sampling positions for a reference search.
POSITION_ATTEMPTS = 20
