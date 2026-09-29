import logging
from collections.abc import Mapping, Sequence

from openmind.inference.model.substitution import Substitution
from openmind.inference.service.evaluable_predicates import EvaluablePredicates
from openmind.inference.service.unifier import Unifier
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


class Subsumer:
    """Whether one clause says everything another says, so the other adds nothing.

    This is what a set of conclusions needs to stop growing. Reasoning forward produces the same thing over and
    over in slightly different dress, and without a way to notice it the set never settles and the search drowns in
    its own output. It is also how things get ordered without looking at anything: where everything one thing's
    rules allow another's allow too, the second affords at least what the first does, in every position, always.

    A clause says everything another says when there is one way of reading its variables that turns every one of
    its literals into a literal the other has. Reading the variables only one way is what makes it an ordering
    rather than a guess: a clause does not get to mean different things in different parts of itself.

    **And one thing more.** Saying a number is at least eight and saying it is at least five are neither
    of them an instance of the other, so nothing about their shape says the second adds nothing once the first is
    had — and yet it adds nothing. So the orderings are consulted, and a conclusion already reached in a stronger
    form is not reached again in a weaker one for ever. The stand-in did this by hand for the one case it knew
    about; here it is asked of every predicate that carries an ordering.

    It keeps nothing: built once, it is given the clauses on every call."""

    def __init__(self, unifier: Unifier | None = None, evaluable: EvaluablePredicates | None = None) -> None:
        self._unifier = Unifier() if unifier is None else unifier
        self._evaluable = EvaluablePredicates() if evaluable is None else evaluable

    def subsumes(self, one: Clause, other: Clause) -> bool:
        """Whether `one` says everything `other` says, so holding `one` makes `other` add nothing.

        The empty clause says everything: there is nothing left to be true, so nothing is worth adding to it."""
        if one.empty:
            return True
        if len(one.literals) > len(other.literals):
            return False
        return self._covers(one.literals, other.literals, Substitution())

    def redundant(self, clause: Clause, held: Sequence[Clause]) -> bool:
        """Whether anything already held makes that clause add nothing."""
        return any(self.subsumes(one, clause) for one in held)

    def within(self, clauses: Sequence[Clause], others: Sequence[Clause]) -> bool:
        """Whether everything those clauses allow, these allow too.

        A set allows what any of its clauses allows, so it takes in another set where every clause of that set is
        said by one of these."""
        return all(any(self.subsumes(one, other) for one in clauses) for other in others)

    def ordered(self, by: Mapping[Value, Sequence[Clause]]) -> tuple[tuple[Value, Value], ...]:
        """Which of those things affords at least what another does, from their clauses alone.

        Every pair where the first takes in the second and the second does not take in the first: an ordering of
        what the things in a game are for, had before a single position is looked at."""
        found = [
            (one, other)
            for one, mine in by.items()
            for other, theirs in by.items()
            if one != other and self.within(mine, theirs) and not self.within(theirs, mine)
        ]
        logger.info("Concluded %d orderings from %d sets of clauses, with no position looked at", len(found), len(by))
        return tuple(found)

    def compatible(self, one: Clause, other: Clause) -> bool:
        """Whether the two bodies could hold of the same thing.

        Two clauses are compatible unless they ask contradictory things: a literal one affirms and the other
        denies, of terms that can be made the same. Anything else is left open, since a thing one clause says
        nothing about is not a thing it denies."""
        for mine in one.literals:
            for theirs in other.literals:
                if mine.opposes(theirs) and self._unifier.unify(mine, theirs) is not None:
                    return False
        return True

    def _covers(self, wanted: Sequence[Literal], held: Sequence[Literal], agreed: Substitution) -> bool:
        """Whether every wanted literal is among the held ones, under one reading of the variables."""
        if not wanted:
            return True
        first, rest = agreed.applied(wanted[0]), wanted[1:]
        for one in held:
            found = self._unifier.matches(first, one)
            if found is not None and self._covers(rest, held, agreed.then(found)):
                return True
            if self._evaluable.dominates(first, one) and self._covers(rest, held, agreed):
                return True
        return False
