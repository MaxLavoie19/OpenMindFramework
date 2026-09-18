DEFAULT_GAMES = 20
DEFAULT_HELD_OUT_GAMES = 5
DEFAULT_ITERATIONS = 200
DEFAULT_SEED = 1
SEED_RANGE = 2**32

#: Value rules fit one row per position and player, and the payoff a single game ends with is a noisy target.
DEFAULT_VALUE_GAMES = 100
DEFAULT_VALUE_HELD_OUT_GAMES = 25
#: Every position of a game valued at the game's final payoff for each player.
OUTCOME_TARGET = "outcome"
#: Every position valued at the search's mean payoff there for the player to act.
SEARCH_TARGET = "search"
VALUE_TARGETS = (OUTCOME_TARGET, SEARCH_TARGET)

#: A value training loop's defaults: rounds, self-play games per round, MCTS iterations, rollout actions played before a
#: position is valued with the previous round's rules, and games against each opponent after every round.
DEFAULT_TRAINING_ROUNDS = 3
DEFAULT_TRAINING_GAMES = 200
DEFAULT_TRAINING_HELD_OUT_GAMES = 50
DEFAULT_TRAINING_ITERATIONS = 100
DEFAULT_TRAINING_ROLLOUT_ACTIONS = 10
DEFAULT_EVALUATION_GAMES = 20
#: How a training loop names the value rules round 1's self-play starts from, as an opponent.
START_RULES = "start rules"
