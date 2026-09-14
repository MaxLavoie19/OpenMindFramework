DEFAULT_MIN_VISITS = 5
DEFAULT_MAX_CONDITIONS = 2
DEFAULT_MIN_RULE_VISITS = 50
DEFAULT_MIN_GAIN = 0.05
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BEAM_WIDTH = 20
DEFAULT_MAX_OFFSET = 2
DEFAULT_SOLO_LIMIT = 2
DEFAULT_PATTERNS = 200
DEFAULT_FALSE_DISCOVERY_RATE = 0.05
DEFAULT_PERMUTATIONS = 10_000

#: States where a hypothesis's matching and other actions both occur, at least, for it to be discovered.
MIN_STATES = 5
#: A rule whose payoff bound is within this share of the payoff range from the best or worst payoff is a priority rule.
PRIORITY_MARGIN = 0.05
#: Distinct values a quantity is cut at, at most; beyond that, cuts fall at evenly spaced quantiles.
QUANTITY_CUTS = 6
#: Rows probed for the variable an action sets, to decide near()'s offsets.
ANCHOR_PROBES = 50
#: Permutations drawn at once, to bound memory.
PERMUTATION_BATCH = 1_000
