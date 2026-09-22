from dataclasses import dataclass

from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number, Variable


@dataclass(frozen=True, slots=True)
class Clause:
    """A disjunction of literals: the one form everything the engine reasons with takes.

    Four things that look different are the same thing here, which is why there is one mechanism and not four.
    A **fact** is one positive literal with no variables. A **rule** is one positive literal and some negative
    ones, read as "the positive holds wherever all the negatives do". A **goal** is negatives with no positive: it
    is a question asked by denying its answer. And the clause with no literals at all is a **contradiction** —
    deriving it is what a proof is.

    `probability` is how often the rule holds where its body does, 1.0 for one that simply holds. A rule below 1 is
    read as the rule together with one independent chance of its own firing, so there is a single semantics rather
    than a separate treatment for the uncertain case. It is also the only way to state a game with chance in it:
    an outcome that comes about four times in five has nowhere else to be written down.

    `name` is what it is called where it has been named, so a derivation can say what it used."""

    literals: tuple[Literal, ...] = ()
    probability: float = 1.0
    name: str = ""

    @property
    def positive(self) -> tuple[Literal, ...]:
        return tuple(one for one in self.literals if not one.negated)

    @property
    def negative(self) -> tuple[Literal, ...]:
        return tuple(one for one in self.literals if one.negated)

    @property
    def definite(self) -> bool:
        """Whether exactly one literal is positive.

        The fragment that reads as a rule with one conclusion, that the fast path and the learner work in, and that
        most of what a game has to say lives in."""
        return len(self.positive) == 1

    @property
    def head(self) -> Literal | None:
        """What it concludes, where it concludes one thing; None for a goal, and where it concludes several."""
        positive = self.positive
        return positive[0] if len(positive) == 1 else None

    @property
    def body(self) -> tuple[Literal, ...]:
        """What has to hold for it to conclude, as the conditions they read as rather than as denials."""
        return tuple(one.denied for one in self.negative)

    @property
    def empty(self) -> bool:
        """The contradiction: nothing is left to be true."""
        return not self.literals

    @property
    def ground(self) -> bool:
        return all(one.ground for one in self.literals)

    @property
    def certain(self) -> bool:
        """Whether it holds whenever its body does, rather than some of the time."""
        return self.probability >= 1.0

    @property
    def variables(self) -> tuple[Variable, ...]:
        found: dict[Variable, None] = {}
        for literal in self.literals:
            for variable in literal.variables:
                found.setdefault(variable, None)
        return tuple(found)

    @property
    def readable(self) -> str:
        """`0.8::worth_at_least(Thing, N) :- reaches(Thing, N)`, the probability left off where it is certain."""
        said = "" if self.certain else f"{self.probability:g}::"
        head, body = self.head, self.body
        if self.empty:
            return f"{said}contradiction"
        if head is None:
            return said + "? " + ", ".join(self._said(one) for one in body)
        if not body:
            return said + self._said(head)
        return f"{said}{self._said(head)} :- " + ", ".join(self._said(one) for one in body)

    def _said(self, literal: Literal) -> str:
        negated = "not " if literal.negated else ""
        if not literal.arguments:
            return f"{negated}{literal.predicate}"
        return f"{negated}{literal.predicate}({', '.join(self._term(one) for one in literal.arguments)})"

    def _term(self, term: object) -> str:
        arguments = getattr(term, "arguments", None)
        if arguments is not None:
            return f"{getattr(term, 'name', '?')}({', '.join(self._term(one) for one in arguments)})"
        if isinstance(term, Constant):
            return "nothing" if term.name is None else str(term.name)
        if isinstance(term, Number):
            return f"{term.value:g}" if isinstance(term.value, float) else str(term.value)
        return str(getattr(term, "name", term))
