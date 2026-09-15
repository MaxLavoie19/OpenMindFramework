from openmind.parallel.constant.parallel_constant import DEFAULT_PROCESS_MEMORY as PROCESS_MEMORY
from openmind.parallel.constant.parallel_constant import HALF_THE_MEMORY

#: The player a position is valued for, and the next player in the domain's order, as generated expressions name them.
ME = "me"
OTHER = "other"
#: What a variable reads at an index the position doesn't have.
OUTSIDE = "<outside>"
#: The name a generated expression reads the position it values through.
HERE = "here"
#: Where an expression's template reads its position: `here` at the top, a look-ahead's lambda variable inside one.
VIEW = "{view}"
#: A look-ahead's lambda variable, numbered by how many actions its body looks ahead: v1, v2, ...
LOOK_AHEAD_VARIABLE = "v"
#: The index variable a pattern counts over.
PATTERN_INDEX = "at"
#: The index variables an aggregate reads its body at: one index, or a pair of different indices.
AGGREGATE_INDEX = "i"
AGGREGATE_OTHER_INDEX = "j"
#: Look-ahead kinds: the highest and lowest expected reading after an action, and how many actions the reading holds after.
BEST = "best"
WORST = "worst"
COUNT = "count"
#: Aggregate kinds: how many indices the body holds at, its sum, its lowest and its highest.
SUM = "sum"
LOWEST = "min"
HIGHEST = "max"
AGGREGATES = (COUNT, SUM, LOWEST, HIGHEST)
#: How an aggregate's body grows by a reading at its indices.
BODY_OPERATIONS = ("+", "-", "*", "/", "abs", ">=", "<=", "==", "and", "or")
#: How a pattern condition compares a variable.
PATTERN_RELATIONS = ("==", "!=")
#: How two expressions combine into one.
COMBINATIONS = ("+", "-", "*", "/", "max", "min", ">=", "==")
#: How an expression changes on its own.
UNARY = ("abs",)
#: How an expression is compared with a value it takes.
THRESHOLDS = (">=", "<=")
#: Views and moves remembered between two checks of the process's memory.
MEMORY_CHECK_INTERVAL = 1_000
#: Share of the training rows a new expression is first evaluated on, before it earns evaluation on every row; every row
#: when there are no more than MIN_SCREENING_ROWS.
SCREENING_SHARE = 0.1
MIN_SCREENING_ROWS = 500
#: Candidates evaluated together, between checks of the time, memory and candidates left.
CANDIDATE_BATCH = 200
#: The name a search gives the only target it is given as an array.
SINGLE_TARGET = "target"


#: How long a search runs by default, in seconds.
DEFAULT_SEARCH_SECONDS = 3600.0
#: How long one position is deduced by default, in seconds, and the highest payoff a player can get by default, as in
#: games paying 1 for a win, 0.5 for a draw and 0 for a loss.
DEFAULT_DEDUCTION_SECONDS = 10.0
DEFAULT_HIGHEST_PAYOFF = 1.0
#: The lowest payoff a player can get by default, as in games paying 0 for a loss.
DEFAULT_LOWEST_PAYOFF = 0.0
#: How many bytes a search's process holds at most by default: half the machine's memory.
DEFAULT_SEARCH_MEMORY = HALF_THE_MEMORY
#: How many bytes a process holds before the mechanics clear their views, until a search sets its own share: half the
#: machine's memory shared between its logical CPUs, so that every process of a pool fits together.
DEFAULT_PROCESS_MEMORY = PROCESS_MEMORY
