from dataclasses import dataclass, field

from openmind.rule.model.term import Functor, Term, Variable


@dataclass(frozen=True, slots=True)
class Literal:
    """One thing said of some terms, or its denial.

    The predicate is a reading's name with its slots opened up. What the readings write as the single key
    `"rows from source to target"` is `rows(source, target, N)` here — the template's slots became arguments
    instead of being filled in and forgotten. That is the whole difference between this and the conditions it
    replaces, and everything a rule can now say that it could not follows from it.

    `negated` denies it. A rule's body is written as denials, since a clause is a disjunction and a condition is
    something whose failing would make the rule not apply."""

    predicate: str
    arguments: tuple[Term, ...] = ()
    negated: bool = False

    #: Worked out once when the literal is made, and never part of what makes two literals the same.
    #:
    #: These are asked for constantly — every match, every scoring of a candidate against a case — and working
    #: them out each time means rebuilding the same tuple millions of times over a single run. A literal cannot
    #: change, so neither can its variables.
    variables: tuple[Variable, ...] = field(init=False, compare=False, hash=False, repr=False, default=())
    ground: bool = field(init=False, compare=False, hash=False, repr=False, default=True)

    #: Worked out once, for the same reason as a term's: a case is a set of these and a clause asks after them one
    #: at a time, so the same literal is hashed over and over and can never have changed in between.
    _hash: int = field(init=False, compare=False, hash=False, repr=False, default=0)

    def __post_init__(self) -> None:
        found: dict[Variable, None] = {}
        for argument in self.arguments:
            for variable in self._variables(argument):
                found.setdefault(variable, None)
        object.__setattr__(self, "variables", tuple(found))
        object.__setattr__(self, "ground", not found)
        object.__setattr__(self, "_hash", hash((self.predicate, self.arguments, self.negated)))

    @property
    def arity(self) -> int:
        return len(self.arguments)

    @property
    def denied(self) -> "Literal":
        """The same thing said the other way."""
        return Literal(self.predicate, self.arguments, not self.negated)

    def opposes(self, other: "Literal") -> bool:
        """Whether the two say the same thing and disagree about it, which is what lets them be resolved."""
        return self.predicate == other.predicate and self.arity == other.arity and self.negated != other.negated

    def _variables(self, term: Term) -> tuple[Variable, ...]:
        if isinstance(term, Variable):
            return (term,)
        if isinstance(term, Functor):
            return tuple(one for argument in term.arguments for one in self._variables(argument))
        return ()


def _hashed(held: Literal) -> int:
    return held._hash


# The dataclass writes a hash of its own for a frozen class, so the cached one is put in afterwards rather than in
# the body, where it would be overwritten.
Literal.__hash__ = _hashed  # type: ignore[assignment,method-assign]
