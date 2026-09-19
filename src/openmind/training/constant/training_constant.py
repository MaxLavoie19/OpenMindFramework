DEFAULT_ITERATIONS = 200
SEED_RANGE = 2**32

#: Every position of a game valued at the game's final payoff for each player.
OUTCOME_TARGET = "outcome"
#: Every position valued at the search's mean payoff there for the player to act.
SEARCH_TARGET = "search"
VALUE_TARGETS = (OUTCOME_TARGET, SEARCH_TARGET)
