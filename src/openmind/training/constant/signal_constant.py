import math

from openmind.rbs.constant.generation_constant import DEFAULT_SOLO_LIMIT

#: The signal of winning itself: a position's payoffs, which always point to the winner.
WIN = "win"
#: The aggregation where every signal it reads votes the same.
UNIFORM = "uniform"
#: The aggregation where every signal it reads votes by its reliability: until arms play each other, the round's own value
#: rules, those the next round's self-play follows, are this signal's.
WEIGHTED = "weighted"
#: How many signals a round follows at most, winning and the aggregations aside.
DEFAULT_ARMS = 8
#: How many plies later a position's signals are read for its target by default: 0 reads them there.
DEFAULT_SIGNAL_HORIZON = 0
#: UCB1's exploration weight when choosing which arms play each other, by default.
DEFAULT_ARM_EXPLORATION = math.sqrt(2)
#: The signals deduced from the rules: the player's legal moves minus the other player's.
OPTIONS = "options"
#: The player's legal moves.
MY_OPTIONS = "my options"
#: Minus the other player's legal moves.
THEIR_OPTIONS = "their options"
#: The other player's distance to a win minus the player's, a distance being the fewest moves to a win if the other
#: player did nothing.
GOAL_DISTANCE = "goal distance"
#: How many moves ahead goal distance looks for a win by default.
DEFAULT_GOAL_LIMIT = DEFAULT_SOLO_LIMIT
#: Followed by a base: the player's things minus the other player's.
OWNED = "owned"
#: Followed by a base: the most of the other player's things the player can take with one move.
TAKING = "taking"
#: Followed by a base: minus the most of the player's things the other player can take with one move.
LOSING = "losing"
#: Followed by a base: the most legal moves the other player loses from one of the player's moves that takes a thing.
TAKING_MOVES = "taking moves"
#: Followed by a base: minus the most legal moves the player loses from one of the other player's moves that takes a thing.
LOSING_MOVES = "losing moves"
#: Followed by a base: how many of the player's moves leave at least two moves each taking one of the other's things.
FORK = "fork"
#: Followed by a base: the worth of the player's things minus the other's, a thing worth its kind's average moves alone.
MATERIAL = "material"
#: Followed by a base: minus how many of the player's things the other player can take and nothing of the player's
#: could take back.
HANGING = "hanging"
#: The value base weighing every deduced signal the same, which round 1's arms start from.
DEDUCED = "deduced"
#: How a variation of the deduced value base is named after the signal whose weight it doubles: `deduced, <signal> doubled`.
DOUBLED = "doubled"
