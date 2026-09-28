from dataclasses import dataclass, field

from openmind.statement.model.term import Constant, Functor, Number, Term, Variable


@dataclass(frozen=True, slots=True)
class Literal:
    """One thing said of some terms, or its denial.

    The predicate is a reading's name with its slots opened up. What the readings write as the single key
    `"rows from source to target"` is `rows(source, target, N)` here — the template's slots became arguments
    instead of being filled in and forgotten. That is the whole difference between this and the conditions it
    replaces, and everything a rule can now say that it could not follows from it.

    `negated` denies it. A rule's body is written as denials, since a clause is a disjunction and a condition is
    something whose failing would make the rule not apply.

    `when` is the moment it is said of, and **None is now**. Every reading there has ever been is of the
    position as it stands, so saying nothing means saying now, and moments arrive without a single existing
    clause having to be rewritten to mention one. What it buys is the thing that had to be special machinery
    before: a king may not be left where it can be taken is one reading held now and denied at a later moment,
    which is a clause rather than a question somebody had to build an answering service for."""

    predicate: str
    arguments: tuple[Term, ...] = ()
    negated: bool = False
    #: When it is said to hold. None is now.
    when: Term | None = None

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
        object.__setattr__(self, "_hash", hash((self.predicate, self.arguments, self.negated, self.when)))

    @property
    def arity(self) -> int:
        return len(self.arguments)

    @property
    def denied(self) -> "Literal":
        """The same thing said the other way, of the same moment."""
        return Literal(self.predicate, self.arguments, not self.negated, self.when)

    @property
    def readable(self) -> str:
        """The literal said as itself: `takes(rook)`, `not holds(1, 2, grid, nothing)`.

        **A literal that cannot name itself is a literal nothing can label a column with.** Saying one lived
        inside `Clause`, privately, because a clause was the only thing that ever printed one — so anything
        else wanting the words had to wrap a single literal in a clause to get at them. The rendering belongs
        to the thing being rendered."""
        negated = "not " if self.negated else ""
        if not self.arguments:
            return f"{negated}{self.predicate}"
        return f"{negated}{self.predicate}({', '.join(_said(one) for one in self.arguments)})"

    def said_of(self, when: Term | None) -> "Literal":
        """The same thing said of that moment."""
        return Literal(self.predicate, self.arguments, self.negated, when)

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


def _said(term: object) -> str:
    """A term as words: a functor with its arguments, a constant by its name, a number plainly."""
    arguments = getattr(term, "arguments", None)
    if arguments is not None:
        return f"{getattr(term, 'name', '?')}({', '.join(_said(one) for one in arguments)})"
    if isinstance(term, Constant):
        return "nothing" if term.name is None else str(term.name)
    if isinstance(term, Number):
        return f"{term.value:g}" if isinstance(term.value, float) else str(term.value)
    return str(getattr(term, "name", term))
