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

#: Being the same thing and being a different thing, which are asked of anything and not only of numbers.
SAME, OTHER_THAN = "same", "other_than"
ALIKE = (SAME, OTHER_THAN)

#: How far apart two numbers are: `apart(A, B, N)` holds where A and B differ by N, either way round.
#:
#: The comparisons can say that two numbers differ and never by how much, and almost everything a game has to say
#: about where a thing may go is a distance. A rook is the one piece in chess whose rule survives without this,
#: and only by accident — "along a row" happens to be sayable as "the same number". Everything else moves by an
#: offset, and without this an offset cannot be said at all: the rules are not hard to find, they are outside the
#: space anything could search.
#:
#: It earns its place on the same terms as the comparisons: it means the same thing in every game. Any game with
#: numbered anything has distances between them. What it is emphatically not is "rows apart" or "on the same
#: diagonal" — those name a board, and they stay where they belong, as clauses a learner builds out of this.
APART = "apart"

#: Those predicates whose last place can be worked out from the others rather than only checked against them.
#:
#: A predicate that can only be checked can say whether two rows are three apart; it cannot say how far apart they
#: are, and so it cannot be used to compare one distance with another. "The rows apart equal the columns apart" —
#: which is a bishop, and a knight and a king and a pawn with different numbers in it — needs the distance to be
#: had, not guessed at and tested. So where the other places are settled, the last one is computed and stands for
#: what it comes to.
COMPUTING: Mapping[str, Callable[..., float]] = {APART: lambda held, wanted: abs(held - wanted)}


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
        built_in[APART] = lambda held, wanted, by: abs(held - wanted) == by
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
        # Being the same thing and being a different thing are questions about anything at all, and the rest are
        # questions about numbers. Asking whether two colours are the same through a number is asking nothing:
        # neither is a number, so there is no answer, and a rule that rests on it quietly never holds.
        if literal.predicate in ALIKE and literal.arity == 2:
            held = self._plain(literal.arguments[0]) == self._plain(literal.arguments[1])
            return (held if literal.predicate == SAME else not held) != literal.negated
        values = [self._value(one) for one in literal.arguments]
        if any(one is None for one in values):
            return None
        try:
            held = bool(answer(*values))
        except (TypeError, ValueError, ArithmeticError):
            return None
        return held != literal.negated

    def computes(self, predicate: str) -> bool:
        """Whether that predicate's last place can be worked out from the others."""
        return predicate in COMPUTING

    def value(self, literal: Literal) -> float | None:
        """What its last place must be for it to hold, from the others — or None where they are not all settled.

        The literal is otherwise left alone. Whether the answer is what the literal already says is the caller's
        question; this only says what the answer is."""
        answer = COMPUTING.get(literal.predicate)
        if answer is None or literal.arity < 2:
            return None
        values = [self._value(one) for one in literal.arguments[:-1]]
        if any(one is None for one in values):
            return None
        try:
            return float(answer(*values))
        except (TypeError, ValueError, ArithmeticError):
            return None

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

    def _plain(self, term: Term) -> object:
        """What a term amounts to for asking whether two things are the same: the thing it names, whatever kind of
        thing that is. A number written as a value and the same number written as a number are one thing."""
        if isinstance(term, Number):
            return term.value
        if isinstance(term, Constant):
            return term.name
        return term

    def _value(self, term: Term) -> float | None:
        if isinstance(term, Number):
            return float(term.value)
        if isinstance(term, Constant) and isinstance(term.name, (int, float)) and not isinstance(term.name, bool):
            return float(term.name)
        return None
