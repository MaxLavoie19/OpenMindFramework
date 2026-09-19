#: What OMF's clocks do while a debug session is paused.
FREEZE = "freeze"
RUN = "run"
CLOCKS_WHILE_PAUSED = (FREEZE, RUN)

#: What the console answers a pause with.
CONTINUE = "continue"
STEP = "step"
OUT = "out"
RESUMING = (CONTINUE, STEP, OUT)
STATE = "state"
STACK = "stack"
BREAKPOINTS = "breakpoints"
HELP = "help"

#: The name the debugger's warnings are kept under in the knowledge base: its mechanism and its context.
DEBUGGER = "debugger"

#: The kind of warning a WARNING log line becomes when nothing more precise is given.
LOGGED = "logged"

#: How a log line is written: when it was produced, then its level, the service that wrote it and what it says. Runs
#: last for hours or days, and a line without its time can't be placed against anything else that happened.
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s %(message)s"
