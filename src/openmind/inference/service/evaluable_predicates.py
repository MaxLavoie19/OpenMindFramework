import logging
import operator
from collections.abc import Callable, Mapping, Sequence

from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number, Term

logger = logging.getLogger(__name__)

#: Comparing two numbers, and what each comparison lets be concluded about another asking less.
#:
#: These are here because every game has numbers in it and deriving that three is less than four, a step at a time,
#: would be absurd. They hold whatever the game is, which is the only thing that earns a place in the engine.
COMPARISONS: Mapping[str, Callable[[float, float], bool]] = {
    "less": operator.lt,
    "at_most": operator.le,
    "more": operator.gt,
    "at_least": operator.ge,
    "same": operator.eq,
    "other_than": operator.ne,
}

#: Which comparisons say something about a thing being large, and which about it being small. A bound of the same
#: sort that is tighter makes a looser one add nothing; bounds of different sorts say nothing about each other.
UPWARD = ("at_least", "more")
DOWNWARD = ("at_most", "less")


class EvaluablePredicates:
    """The literals answered by computing rather than by deriving.

    Arithmetic and the comparisons are built in because they mean the same thing in every game. **Anything narrower
    is registered, not built in.** A game that lays its things out on a grid has cheap ways of reading it that mean
    nothing elsewhere, and those arrive through `register` from whatever knows the game has one — so a game with no
    grid never sees them, and nothing in the engine has to ask whether there is a grid at all.

    It keeps nothing and registering gives back another of these rather than changing this one, so a game's own
    predicates travel with the game and never leak into another game's reasoning."""

    def __init__(self, answering: Mapping[str, Callable[..., object]] | None = None,
                 orderings: Mapping[str, str] | None = None) -> None:
        built_in: dict[str, Callable[..., object]] = {
            name: (lambda held, wanted, compare=compare: compare(held, wanted)) for name, compare in COMPARISONS.items()
        }
        self._answering: dict[str, Callable[..., object]] = {**built_in, **(answering or {})}
        self._orderings: dict[str, str] = {name: name for name in COMPARISONS} | dict(orderings or {})

    def evaluable(self, predicate: str) -> bool:
        """Whether that predicate is answered by computing."""
        return predicate in self._answering

    def holds(self, literal: Literal) -> bool | None:
        """Whether it holds; None where it is not evaluable, or not settled enough to say.

        A literal still holding a variable is not false, it is unanswered — which is a different thing, and saying
        so is what lets the reasoning come back to it once the variable stands for something."""
        answer = self._answering.get(literal.predicate)
        if answer is None or not literal.ground:
            return None
        values = [self._value(one) for one in literal.arguments]
        if any(one is None for one in values):
            return None
        try:
            held = bool(answer(*values))
        except (TypeError, ValueError, ArithmeticError):
            return None
        return held != literal.negated

    def dominates(self, one: Literal, other: Literal) -> bool:
        """Whether holding the first makes the second add nothing.

        This is what plain subsumption cannot see. Saying a thing is worth at least eight and saying it is worth at
        least five are neither of them an instance of the other, so nothing about their shape says the second is
        pointless once the first is had — and yet it is. What decides it is the ordering the comparison belongs to
        and which bound is tighter, so a conclusion already reached in a stronger form is not reached again in a
        weaker one, over and over, for ever."""
        if one.predicate != other.predicate or one.negated != other.negated or one.arity != other.arity:
            return False
        if one.predicate not in self._orderings or one.arity != 2:
            return False
        held, wanted = self._value(one.arguments[1]), self._value(other.arguments[1])
        if held is None or wanted is None or one.arguments[0] != other.arguments[0]:
            return False
        if one.predicate in UPWARD:
            return held >= wanted
        if one.predicate in DOWNWARD:
            return held <= wanted
        return held == wanted

    def register(
        self, predicate: str, answering: Callable[..., object], ordering: str = ""
    ) -> "EvaluablePredicates":
        """The same predicates with one more, for a game that has something worth computing."""
        logger.debug("Registered %s as computed%s", predicate, f", ordered by {ordering}" if ordering else "")
        return EvaluablePredicates(
            {**self._answering, predicate: answering},
            {**self._orderings, **({predicate: ordering} if ordering else {})},
        )

    def registered(self, predicates: Sequence[str] = ()) -> tuple[str, ...]:
        """Every predicate answered by computing, or those of the given ones that are."""
        held = tuple(self._answering)
        return tuple(one for one in predicates if one in self._answering) if predicates else held

    def _value(self, term: Term) -> float | None:
        if isinstance(term, Number):
            return float(term.value)
        if isinstance(term, Constant) and isinstance(term.name, (int, float)) and not isinstance(term.name, bool):
            return float(term.name)
        return None
