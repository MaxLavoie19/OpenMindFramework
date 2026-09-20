#: The tasks OMF knows, each named here and each with a port of its own. A ruleset says which task it is a model of,
#: and so does a model record. A task whose port has no model yet gets its port at the step that builds it.
#:
#: Modelling another agent is not a task of its own: an agent model is the position value and move value heuristics
#: trained to predict that agent's play, registered in that agent's context.
SIMULATION = "simulation"
PREDICTION = "prediction"
POSITION_VALUE = "position value"
MOVE_VALUE = "move value"
POLICY_VALUE = "policy value"
POLICY_PICKING = "policy picking"
OPTIMIZING = "optimizing"
INFERENCE = "inference"
BINNING = "binning"
ABSTRACTION = "abstraction"
HYPOTHESIS = "hypothesis"
PLANNING = "planning"
TIME_MANAGEMENT = "time management"

#: Every task, in the order they are named.
TASKS = (
    SIMULATION,
    PREDICTION,
    POSITION_VALUE,
    MOVE_VALUE,
    POLICY_VALUE,
    POLICY_PICKING,
    OPTIMIZING,
    INFERENCE,
    BINNING,
    ABSTRACTION,
    HYPOTHESIS,
    PLANNING,
    TIME_MANAGEMENT,
)
