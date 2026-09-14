DEFAULT_PAIR_POOL = 20
DEFAULT_CUTS = 6
#: Swept from the highest price down, each fit starting from the previous one's weights.
DEFAULT_PRICES = (0.1, 0.03, 0.01, 0.003, 0.001)
DEFAULT_MAX_STEPS = 1000
DEFAULT_TOLERANCE = 1e-6

#: The name a count term gives each value it counts: sum(value == me for value in cell.values()).
COUNT_VARIABLE = "value"
#: A variable, or an indexed variable's base, with more distinct values than this, such as a history of positions or a
#: move clock, gets no term per value: each term would match a handful of rows.
MAX_VALUE_TERMS = 32
