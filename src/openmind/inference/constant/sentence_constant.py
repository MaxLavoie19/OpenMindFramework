#: How a literal reading names a player: possessive and subject.
PLAYER_WORDS = {"me": ("my", "I"), "other": ("the opponent's", "the opponent")}
#: How a literal reading names a player as a value: `the color there is mine`.
PLAYER_VALUES = {"me": "mine", "other": "the opponent's"}
#: How a literal reading names a place a pattern counts, and the entries an aggregate reads.
PLACE = "there"
ONE_ENTRY = "one entry"
THAT_ENTRY = "that entry"
OTHER_ENTRY = "the other entry"
#: Words for values.
EMPTY_WORD = "empty"
OUTSIDE_WORD = "outside"
NOW_WORD = "now"
#: Comparisons, arithmetic and connectives.
COMPARISON_WORDS = {
    "Eq": "is",
    "NotEq": "is not",
    "GtE": "is at least",
    "LtE": "is at most",
    "Gt": "is more than",
    "Lt": "is less than",
}
ARITHMETIC_WORDS = {"Add": "plus", "Sub": "minus", "Mult": "times", "Div": "divided by"}
CONNECTIVE_WORDS = {"And": "and", "Or": "or"}
#: Look-aheads: the highest and lowest reading over a player's moves, and how many moves it holds after.
LOOK_AHEAD_TEMPLATES = {
    "best": "the highest, over {possessive} moves, of {body}",
    "worst": "the lowest, over {possessive} moves, of {body}",
    "count": "the number of {possessive} moves after which {body}",
}
#: A position read on an outer view inside a look-ahead.
AFTER_MOVE = "after {possessive} move"
THEN_MOVE = "{before}, then {possessive} move"
#: Aggregates over the entries of a base: what they run over, and what a count counts.
EVERY_ENTRY = "every entry"
EVERY_PAIR = "every pair of different entries"
ENTRIES = "entries"
PAIRS = "pairs of different entries"
AGGREGATE_TEMPLATES = {
    "count": "the number of {entries} where {body}",
    "sum": "the total, over {entries}, of {body}",
    "min": "the lowest, over {entries}, of {body}",
    "max": "the highest, over {entries}, of {body}",
}
PATTERN_TEMPLATE = "the number of places where {body}"
#: Other constructs.
CHANGE_TEMPLATE = "the change in {body}"
WHETHER_TEMPLATE = "whether {body}"
DISTANCE_TEMPLATE = "the distance between {first} and {second}"
SIZE_TEMPLATE = "the size of {body}"
LARGER_TEMPLATE = "the larger of {first} and {second}"
SMALLER_TEMPLATE = "the smaller of {first} and {second}"
AT_LEAST_ONE_TEMPLATE = "{body}, counted as at least 1"
MOBILITY_TEMPLATE = "the number of moves {subject} could make"
OFFSET_TEMPLATE = "the {base} {steps} away from {place}"
READING_TEMPLATE = "the {base}"
AT_TEMPLATE = "the {base} at {indices}"
OF_TEMPLATE = "the {base} of {entry}"
OWNED_TEMPLATE = "{possessive} {base}"
QUALIFIED_TEMPLATE = "{reading} {view}"
NOT_TEMPLATE = "not ({body})"
NEGATIVE_TEMPLATE = "minus {body}"
WINS_TEMPLATE = "how many winning moves {subject} would have"
WIN_CHANCE_TEMPLATE = "the chance that this action wins"
#: A construct without a template, quoted as its source.
SOURCE_TEMPLATE = "`{source}`"
