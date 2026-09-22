import logging

from openmind.inference.model.substitution import Substitution
from openmind.inference.service.evaluable_predicates import EvaluablePredicates
from openmind.inference.service.unifier import Unifier
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal

logger = logging.getLogger(__name__)


class Resolver:
    """One step of reasoning, and the only one there is.

    Two clauses, one saying something and the other denying it, of terms that can be made to agree: strike the two
    out and what is left of both is a clause that follows from them. That is the whole mechanism, and everything
    the engine does is some number of these.

    What makes it worth building on is that the same step does four jobs that look unrelated. A rule against facts
    gives a **new fact**. A rule against a rule gives a **new rule** — which is what makes something out of two
    things already known, rather than just fetching what was stored. A denied question against the rules, driven
    toward a contradiction, gives a **proof**. And a rule against a case it gets wrong gives a **refutation**.
    There is no second mechanism to keep honest against the first.

    Factoring is the other half. A clause that says the same thing twice of terms that can agree says it once, and
    without collapsing those a general clause set can chase its own tail. A definite clause never needs it, so a
    game's own rules mostly never pay for it.

    It keeps nothing: built once, it is given the clauses on every call."""

    def __init__(self, unifier: Unifier | None = None, evaluable: EvaluablePredicates | None = None) -> None:
        self._unifier = Unifier() if unifier is None else unifier
        self._evaluable = EvaluablePredicates() if evaluable is None else evaluable

    def resolve(self, one: Clause, other: Clause, apart: int = 0) -> tuple[tuple[Clause, Substitution], ...]:
        """Every clause that follows from putting those two together, each with what it had to take things to mean.

        The two are renamed apart first. Two clauses that happen to use the same letter are not talking about the
        same thing, and putting them together without saying so makes them agree about something neither said."""
        mine = self._unifier.renamed(one, apart * 2 + 1)
        theirs = self._unifier.renamed(other, apart * 2 + 2)
        found: list[tuple[Clause, Substitution]] = []
        for first in mine.literals:
            for second in theirs.literals:
                if not first.opposes(second):
                    continue
                agreed = self._unifier.unify(first, second)
                if agreed is None:
                    continue
                kept = [one for one in mine.literals if one != first]
                kept.extend(one for one in theirs.literals if one != second)
                literals = self._distinct(tuple(agreed.applied(one) for one in kept))
                if self._trivial(literals):
                    continue
                found.append((Clause(literals, mine.probability * theirs.probability), agreed))
        return tuple(found)

    def factors(self, clause: Clause) -> tuple[tuple[Clause, Substitution], ...]:
        """The clause with two of its own literals made one, wherever two of them can be made to agree."""
        found: list[tuple[Clause, Substitution]] = []
        for number, first in enumerate(clause.literals):
            for second in clause.literals[number + 1 :]:
                if first.negated != second.negated:
                    continue
                agreed = self._unifier.unify(first, second)
                if agreed is None:
                    continue
                literals = self._distinct(tuple(agreed.applied(one) for one in clause.literals))
                if len(literals) < len(clause.literals):
                    found.append((Clause(literals, clause.probability, clause.name), agreed))
        return tuple(found)

    def settled(self, clause: Clause) -> Clause | None:
        """The clause with every computed literal worked out, or None where the clause has become idle.

        A literal the engine can compute is not something to reason about: comparing two numbers is settled by
        comparing them, not derived a step at a time. A clause is a disjunction, so a computed literal that **holds**
        makes the whole clause true whatever else happens — it says nothing and is dropped, which is `None`. One
        that **fails** can never be what makes the clause true, so it is struck out and the rest carries on. Read as
        a rule, that is exactly right: a condition met is discharged, and a condition failed makes the rule idle
        here. A literal not yet ground enough to settle is left alone, to be settled once something binds it."""
        literals: list[Literal] = []
        for literal in clause.literals:
            if not self._evaluable.evaluable(literal.predicate):
                literals.append(literal)
                continue
            held = self._evaluable.holds(literal)
            if held is None:
                literals.append(literal)
            elif held:
                return None
        return Clause(tuple(literals), clause.probability, clause.name)

    def _distinct(self, literals: tuple[Literal, ...]) -> tuple[Literal, ...]:
        found: list[Literal] = []
        for literal in literals:
            if literal not in found:
                found.append(literal)
        return tuple(found)

    def _trivial(self, literals: tuple[Literal, ...]) -> bool:
        """Whether the clause is true whatever happens, so following it further says nothing."""
        return any(one.denied in literals for one in literals)
