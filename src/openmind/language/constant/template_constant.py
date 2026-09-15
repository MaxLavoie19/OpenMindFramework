#: The template encoder's name.
TEMPLATE_ENCODER = "templates"
#: How templates name a player: possessive, subject, and as a value.
POSSESSIVES = {"me": "my", "opponent": "the opponent's"}
SUBJECTS = {"me": "I", "opponent": "the opponent"}
OWNERSHIP = {"me": "mine", "opponent": "the opponent's"}
#: Words for values, places and entries.
EMPTY_WORD = "empty"
OUTSIDE_WORD = "outside"
NOW_WORD = "now"
PLACE_WORDS = {"there": "there"}
ENTRY_WORDS = {"that": "that entry", "one": "one entry", "other": "the other entry"}
#: Comparisons, arithmetic and connectives.
COMPARISON_WORDS = {
    "equals": "is",
    "differs from": "is not",
    "at least": "is at least",
    "at most": "is at most",
    "more than": "is more than",
    "less than": "is less than",
}
CONNECTIVE_WORDS = {"all of": "and", "any of": "or"}
#: Look-aheads.
LOOK_AHEAD_TEMPLATES = {
    "highest": "the highest, over {possessive} moves, of {body}",
    "lowest": "the lowest, over {possessive} moves, of {body}",
    "count": "the number of {possessive} moves after which {body}",
}
AFTER_MOVE = "after {possessive} move"
THEN_MOVE = "{before}, then {possessive} move"
#: Aggregates and pattern counts.
AGGREGATE_TEMPLATES = {
    "count": "the number of {entries} where {body}",
    "total": "the total, over {every}, of {body}",
    "lowest": "the lowest, over {every}, of {body}",
    "highest": "the highest, over {every}, of {body}",
}
EVERY_WORDS = {"entries": "every entry", "pairs of different entries": "every pair of different entries"}
PLACES_TEMPLATE = "the number of places where {body}"
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
PLACE_READING_TEMPLATE = "the {base} {place}"
OF_TEMPLATE = "the {base} of {entry}"
OWNED_TEMPLATE = "{possessive} {base}"
SOURCE_INDEX_TEMPLATE = "the {base} at `{source}`"
QUALIFIED_TEMPLATE = "{reading} {position}"
NOT_TEMPLATE = "not ({body})"
NEGATIVE_TEMPLATE = "minus {body}"
WINS_TEMPLATE = "how many winning moves {subject} would have"
SOLO_DISTANCE_TEMPLATE = "how many of {possessive} own moves away a win is"
WIN_CHANCE_TEMPLATE = "the chance that this action wins"
SOURCE_TEMPLATE = "`{source}`"
#: A rule: its effect, then what it measures.
RULE_TEMPLATES = {"raises": "Raises my value: {measure}.", "lowers": "Lowers my value: {measure}."}
