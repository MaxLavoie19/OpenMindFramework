#: The tasks OMF knows, each named here and each with a port of its own. A ruleset says which task it is a model of,
#: and so does a model record. A task whose port has no model yet gets its port at the step that builds it.
SIMULATION = "simulation"
PREDICTION = "prediction"
POSITION_VALUE = "position value"
MOVE_VALUE = "move value"
TACTIC_VALUE = "tactic value"
INFERENCE = "inference"
BINNING = "binning"
ABSTRACTION = "abstraction"
AGENT_MODEL = "agent model"
PLANNING = "planning"
TIME_MANAGEMENT = "time management"

#: Every task, in the order they are named.
TASKS = (
    SIMULATION,
    PREDICTION,
    POSITION_VALUE,
    MOVE_VALUE,
    TACTIC_VALUE,
    INFERENCE,
    BINNING,
    ABSTRACTION,
    AGENT_MODEL,
    PLANNING,
    TIME_MANAGEMENT,
)
