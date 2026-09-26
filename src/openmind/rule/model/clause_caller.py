from collections.abc import Mapping
from typing import Protocol

from openmind.rule.model.clause import Clause
from openmind.structure.model.value import Value
from openmind.world.model.state import State


class ClauseCaller(Protocol):
    """What answers a learned clause where a rule is asked for: whether it holds of that state and those parameters.

    **`ClauseRule` has always promised a compiler and this is the other answer.** Its docstring says the logic is
    what gets stored and *"the compiler hands an RBS the Python it runs when it needs to run it"*. No such compiler
    was ever written, so everything OMF induced was stored as a rule nothing could run — declared, kept, reasoned
    over, and never once consulted about whether a move was legal.

    Compiling is not the only way across, and it is the worse one. Asking whether a clause refuses a candidate is
    the same question learning asks of it, so one mechanism can serve both and a constraint cannot come to mean one
    thing while it is being learned and another once it is used. A compiler would be a second answer to the same
    question, and the two would drift.

    **What fills this has to read a position the way the learning did.** A clause says what it says over the
    readings it was learned in; read back in a different vocabulary it is a different clause, or nothing at all.
    Nothing here can check that — it is the caller's to keep true by asking the same deductions it learned from."""

    def holds(self, clause: Clause, state: State, parameters: Mapping[str, Value]) -> bool: ...
