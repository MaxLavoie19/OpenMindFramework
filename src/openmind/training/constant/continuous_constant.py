#: How far each weight moves toward what a finished game showed, unless told otherwise.
DEFAULT_LEARNING_RATE = 0.01
#: How long the rule search after a decisive game runs at most, unless told otherwise.
DEFAULT_RULE_SEARCH_SECONDS = 600.0
#: The kind of game continuous training remembers: every game is between two arms.
CONTINUOUS_GAME = "arms"
#: The keywords of what a game's study proved: a position's proven payoffs, and a seed a proof induced.
PROOF_KEYWORD, SEED_KEYWORD = "proof", "seed"
