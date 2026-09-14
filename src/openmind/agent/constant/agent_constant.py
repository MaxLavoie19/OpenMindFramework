import math

DEFAULT_ITERATIONS = 1000
EXPLORATION = math.sqrt(2)
PRIOR_WEIGHT = 1.0
ROLLOUT_TEMPERATURE = 0.2
GUIDED_ROLLOUTS = True
#: Each player's payoff for a rollout stopped at a rollout limit, by default: a draw in games paying 1, 0.5 and 0.
DEFAULT_UNFINISHED_PAYOFF = 0.5
#: The entry point group under which installed projects register their domains' recipes.
DOMAIN_ENTRY_POINTS = "openmind.domains"
