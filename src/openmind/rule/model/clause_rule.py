from dataclasses import dataclass

from openmind.rule.model.clause import Clause


@dataclass(frozen=True, slots=True)
class ClauseRule:
    """A rule the agent can reason with, and not only run.

    A rule kept as Python is opaque the moment it is stored: it can be called, and that is all. Nothing can ask
    whether it says more than another rule, resolve it against a second rule to get a third, look for the position
    that would break it, or read it out in words. Kept as a clause it is open to all of those, and the compiler
    hands an RBS the Python it runs when it needs to run it.

    So what is stored is the logic, and the Python is made from it — rather than the other way round, where the
    logic is lost as soon as it is written down."""

    clause: Clause

    @property
    def readable(self) -> str:
        return self.clause.readable
