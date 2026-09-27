from dataclasses import dataclass, field

from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Functor, Term, Variable


@dataclass(frozen=True, slots=True)
class Substitution:
    """What variables were taken to stand for, on the way to a conclusion.

    Every step of reasoning that puts two rules together has to agree on what their variables mean, and what they
    agreed is as much a part of the conclusion as the conclusion is: a rule about any thing at all, used of one
    particular thing, says something about that thing only because the agreement said so. Kept, it is what lets a
    derivation be read back, and what lets a general rule be used without its ceasing to be general."""

    bindings: tuple[tuple[Variable, Term], ...] = ()

    #: Which variables it says anything about, worked out once: asked for on every application, and an agreement
    #: cannot change.
    variables: frozenset[Variable] = field(init=False, compare=False, hash=False, repr=False, default=frozenset())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", frozenset(held for held, _ in self.bindings))

    def of(self, variable: Variable) -> Term | None:
        """What that variable stands for here, or None where nothing was agreed about it."""
        for held, term in self.bindings:
            if held == variable:
                return term
        return None

    def bound(self, variable: Variable, term: Term) -> "Substitution":
        """The same agreement with one more in it, the new one applied to what was already agreed.

        Applying it as it goes is what keeps the agreement flat: nothing later has to follow a chain of variables
        standing for variables to find out what something means."""
        one = Substitution(((variable, term),))
        kept = tuple((held, one.applied(value)) for held, value in self.bindings)
        return Substitution((*kept, (variable, term)))

    def then(self, other: "Substitution") -> "Substitution":
        """This agreement followed by another: what each variable stands for once both have been applied."""
        kept = tuple((held, other.applied(term)) for held, term in self.bindings)
        added = tuple(one for one in other.bindings if self.of(one[0]) is None)
        return Substitution((*kept, *added))

    def applied[T: (Term, Literal, Clause)](self, node: T) -> T:
        """That term, literal or clause with every variable replaced by what it stands for.

        An agreement about nothing, and an agreement about no variable the thing reads, leave it as it was. Both
        are the common case by far — most candidates speak of variables nothing has been agreed about yet — and
        rebuilding the whole thing to arrive back where it started is where a great deal of time was going."""
        if not self.bindings:
            return node
        if isinstance(node, Literal):
            if all(one not in self.variables for one in node.variables):
                return node  # type: ignore[return-value]
            return Literal(node.predicate, tuple(self.applied(one) for one in node.arguments), node.negated)  # type: ignore[return-value]
        if isinstance(node, Variable):
            return self.of(node) or node  # type: ignore[return-value]
        if isinstance(node, Functor):
            return Functor(node.name, tuple(self.applied(one) for one in node.arguments))  # type: ignore[return-value]
        if isinstance(node, Clause):
            return Clause(tuple(self.applied(one) for one in node.literals), node.probability, node.name)  # type: ignore[return-value]
        return node

    @property
    def readable(self) -> str:
        return ", ".join(f"{variable.name} = {getattr(term, 'name', getattr(term, 'value', term))}" for variable, term in self.bindings) or "nothing bound"
