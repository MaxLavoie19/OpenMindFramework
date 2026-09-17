#: The share of uniform choice mixed into regret matching where players act at once, so every action keeps being
#: sampled (γ in Lanctot, Lisý and Winands, 2013).
DEFAULT_REGRET_EXPLORATION = 0.1

#: How a search picks the action to follow at a node it has tried: UCB1, every legal action tried once first, or PUCT,
#: following Q + c · P · √N / (1 + n) over every legal action with a prior P.
UCB1, PUCT = "ucb1", "puct"
SELECTIONS = (UCB1, PUCT)
#: PUCT's exploration weight c, unless told otherwise.
DEFAULT_PUCT_EXPLORATION = 1.5
#: The temperature of the softmax turning ratings or values into a prior, unless told otherwise.
DEFAULT_PRIOR_TEMPERATURE = 0.1
#: The priors an entry point can name: every action alike, the agent's rules' ratings, or its value rules' values.
UNIFORM_PRIOR, RATER_PRIOR, VALUE_PRIOR = "uniform", "rater", "value"
PRIORS = (UNIFORM_PRIOR, RATER_PRIOR, VALUE_PRIOR)
