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

#: Selection compares guided searches at this budget by default, where guidance mattered most on tic-tac-toe.
DEFAULT_SELECTION_ITERATIONS = 10
#: The largest rise in mean regret a removed rule may cause, by default.
DEFAULT_MARGIN = 0.005
DEFAULT_SELECTION_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
#: Bootstrap resamples drawn at once, to bound memory.
BOOTSTRAP_BATCH = 1_000
#: The confirmation searches with the selection's seed plus this.
CONFIRMATION_SEED_OFFSET = 1
#: How the agent measured during selection is named in the logs.
SELECTION_KIND = "Selection"
