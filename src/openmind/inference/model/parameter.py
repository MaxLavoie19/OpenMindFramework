from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Parameter:
    """A number in a heuristic that the engine reasoned a starting value for, and that the search may move.

    A thing whose rules admit fourteen ways of moving is not thereby worth fourteen. It is worth looking for its
    value near fourteen rather than near nothing, which beats starting at zero and beats a number somebody typed
    in. The engine says fourteen because the rules say fourteen; the games say what it should be.

    `holds` names the ground clause carrying the value, so tuning the parameter is revising that clause and the
    revision stays readable as logic rather than disappearing into a weight vector."""

    name: str
    initial: float
    holds: str = ""

    @property
    def readable(self) -> str:
        return f"{self.name} = {self.initial:g}"
