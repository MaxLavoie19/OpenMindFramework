#: What kinds of model OMF knows of. Any other family is named by whoever registers it.
RULES = "rules"
LOOKUP_TABLE = "lookup table"
DECISION_TREE = "decision tree"
ENSEMBLE = "ensemble"
NETWORK = "network"

#: What the belief timing a model keeps is called, by model id, and the tags it keeps its counts under.
PROCESSING_TIME = "processing time of {model}"
READINGS = "readings"
TOTAL_SECONDS = "total seconds"

#: How far a model that has hardly been measured is preferred over one measured and found wanting.
#:
#: The usual constant of the bound, kept rather than chosen: it is what makes the regret of a whole run grow
#: like the log of its length, which is the property the bound is for. Shared by what draws a model to play
#: and by what decides a retired one is worth asking again, because they are the same question — what is worth
#: trying, given what is known and how little of it there is.
CURIOSITY = 1.4142135623730951
