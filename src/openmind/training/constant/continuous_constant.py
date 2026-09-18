import math

#: The kind of game continuous training remembers: every game is between two arms.
CONTINUOUS_GAME = "arms"
#: The keyword of a position a game's walk back proved.
PROOF_KEYWORD = "proof"
#: The arm taking a seat no value base fills: an agent without value rules.
NO_VALUE_RULES = "no value rules"
#: UCB1's exploration weight when choosing which arms play each other, by default.
DEFAULT_ARM_EXPLORATION = math.sqrt(2)
