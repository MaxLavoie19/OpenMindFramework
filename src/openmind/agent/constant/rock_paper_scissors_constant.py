NAME = "rockpaperscissors"
PLAYERS = ("A", "B")
UNSET = None
CERTAIN = 1.0

ROCK = "rock"
PAPER = "paper"
SCISSORS = "scissors"
SHAPES = (ROCK, PAPER, SCISSORS)
#: Each shape and the shape it beats.
BEATS = {ROCK: SCISSORS, PAPER: ROCK, SCISSORS: PAPER}

#: The payoffs of a win, a draw and a loss.
WIN = 1.0
DRAW = 0.5
LOSS = 0.0

TURN = "turn"
HAND = "hand"
PAYOFF = "payoff"
THROW = "throw"
SHAPE = "shape"
