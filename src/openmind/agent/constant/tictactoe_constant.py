from dataclasses import replace

from openmind.agent.model.tictactoe_variant import TicTacToeVariant

NAME = "tictactoe"
SEPARATOR = "/"
PLAYERS = ("X", "O")
EMPTY = None
UNSET = None

WIN = 1.0
DRAW = 0.5
LOSS = 0.0
CERTAIN = 1.0

CELL = "cell"
TURN = "turn"
PAYOFF = "payoff"
PLACE = "place"
DROP = "drop"
ROW = "row"
COL = "col"

STANDARD = TicTacToeVariant("standard", width=3, height=3, line=3, gravity=False)
VARIANTS = {
    variant.name: variant
    for variant in (
        STANDARD,
        replace(STANDARD, name="fourinarow", width=7, height=6, line=4, gravity=True),
        replace(STANDARD, name="gomoku", width=15, height=15, line=5),
    )
}
