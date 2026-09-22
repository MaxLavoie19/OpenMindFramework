from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Signature:
    """What can be said in a context: every predicate by name and arity, and what its arguments were seen holding.

    The learner's candidate literals come from here and from nowhere else, so what OMF can learn to say is exactly
    what the game's own readings let it say. Nothing is listed that the game did not offer, and nothing the game
    offered is left out because it looked unpromising.

    `evaluable` names the predicates answered by computing rather than by deriving. `values` is what was seen at
    each position of each predicate, which is what a literal comparing to a value can compare to; `numeric` marks
    the positions where every value seen was a number, and those are read as numbers and never compared for
    equality against a value seen."""

    arities: tuple[tuple[str, int], ...] = ()
    evaluable: tuple[str, ...] = ()
    values: tuple[tuple[str, int, tuple[Value, ...]], ...] = ()
    numeric: tuple[tuple[str, int], ...] = ()

    @property
    def predicates(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.arities)

    def arity(self, predicate: str) -> int | None:
        for name, held in self.arities:
            if name == predicate:
                return held
        return None

    def seen(self, predicate: str, position: int) -> tuple[Value, ...]:
        """The values that position was seen holding."""
        for name, held, values in self.values:
            if name == predicate and held == position:
                return values
        return ()

    def is_numeric(self, predicate: str, position: int) -> bool:
        return (predicate, position) in self.numeric

    def is_evaluable(self, predicate: str) -> bool:
        return predicate in self.evaluable
