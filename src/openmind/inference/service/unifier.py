import logging

from openmind.inference.model.substitution import Substitution
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Term, Variable

logger = logging.getLogger(__name__)


class Unifier:
    """What two things said of terms would have to agree on to be the same thing said, or nothing where they can't be.

    Every step of reasoning rests on this. Two rules can be put together only where a literal of one and a literal
    of the other can be made to say the same thing, and what makes them say it is an agreement about what their
    variables stand for. The agreement found here is the most general one there is: it binds what has to be bound
    and nothing more, so a conclusion drawn through it stays as general as it can be. Binding more would be
    guessing, and the conclusion would be about less than it should.

    The occurs check is done. A variable standing for something that contains that same variable describes nothing
    finite, and a prover that allows it builds terms forever on the way to answers it will never give. It costs a
    walk through the term each time and buys termination.

    It keeps nothing: built once, it is given what to unify on every call."""

    def unify(self, one: Literal, other: Literal) -> Substitution | None:
        """What the two would have to agree on to be the same literal, or None.

        They must say the same thing of the same number of terms. Whether one denies it is not looked at here: a
        step that puts two clauses together wants literals that oppose each other, and one that checks whether a
        clause says more than another wants literals that agree, so which is wanted belongs to the caller."""
        if one.predicate != other.predicate or one.arity != other.arity:
            return None
        return self.unify_terms(one.arguments, other.arguments)

    def unify_terms(
        self, one: Term | tuple[Term, ...], other: Term | tuple[Term, ...], agreed: Substitution = Substitution()
    ) -> Substitution | None:
        """The agreement extending the one given, or None where no agreement extends it."""
        if isinstance(one, tuple) or isinstance(other, tuple):
            if not isinstance(one, tuple) or not isinstance(other, tuple) or len(one) != len(other):
                return None
            found: Substitution | None = agreed
            for mine, theirs in zip(one, other):
                if found is None:
                    return None
                found = self.unify_terms(mine, theirs, found)
            return found
        held, wanted = agreed.applied(one), agreed.applied(other)
        if held == wanted:
            return agreed
        if isinstance(held, Variable):
            return None if self._occurs(held, wanted) else agreed.bound(held, wanted)
        if isinstance(wanted, Variable):
            return None if self._occurs(wanted, held) else agreed.bound(wanted, held)
        if isinstance(held, Functor) and isinstance(wanted, Functor):
            if held.name != wanted.name or len(held.arguments) != len(wanted.arguments):
                return None
            return self.unify_terms(held.arguments, wanted.arguments, agreed)
        return None

    def matches(self, general: Literal, particular: Literal) -> Substitution | None:
        """What makes the first into the second, binding only the first's variables, or None.

        One-way, unlike unifying: asking whether a rule covers a case must not go the other way and change the
        case to suit the rule. This is what deciding whether one clause says everything another says is built on.

        Denial has to agree here, where unifying ignores it. Unifying is looking for two literals that can be
        played off against each other, and those are the ones that disagree; matching is asking whether one
        literal is a case of another, and a thing denied is not a case of that thing asserted."""
        if general.predicate != particular.predicate or general.arity != particular.arity:
            return None
        if general.negated != particular.negated:
            return None
        agreed = Substitution()
        for mine, theirs in zip(general.arguments, particular.arguments):
            found = self._match(mine, theirs, agreed)
            if found is None:
                return None
            agreed = found
        return agreed

    def renamed(self, clause: Clause, apart: int) -> Clause:
        """The clause with its variables renamed so none of them is one of anybody else's.

        Two clauses that both happen to call a variable the same thing are not talking about the same thing, and
        putting them together without saying so makes them agree about something neither of them said."""
        agreed = Substitution()
        for variable in clause.variables:
            agreed = agreed.bound(variable, Variable(f"{variable.name}#{apart}", variable.sort))
        return agreed.applied(clause)

    def _match(self, general: Term, particular: Term, agreed: Substitution) -> Substitution | None:
        if isinstance(general, Variable):
            held = agreed.of(general)
            if held is None:
                return agreed.bound(general, particular)
            return agreed if held == particular else None
        if isinstance(general, Functor):
            if not isinstance(particular, Functor) or general.name != particular.name:
                return None
            if len(general.arguments) != len(particular.arguments):
                return None
            found: Substitution | None = agreed
            for mine, theirs in zip(general.arguments, particular.arguments):
                if found is None:
                    return None
                found = self._match(mine, theirs, found)
            return found
        if isinstance(general, (Constant, Number)):
            return agreed if general == particular else None
        return None

    def _occurs(self, variable: Variable, term: Term) -> bool:
        """Whether the variable is inside that term, which would make it stand for something containing itself."""
        if isinstance(term, Variable):
            return term == variable
        if isinstance(term, Functor):
            return any(self._occurs(variable, one) for one in term.arguments)
        return False
