import itertools
import logging

from openmind.inference.model.formula import And, Atom, Exists, ForAll, Formula, Iff, Implies, Not, Or, Truth
from openmind.inference.model.substitution import Substitution
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Functor, Term, Variable

logger = logging.getLogger(__name__)

#: How a thing conjured up to stand for "there is some such thing" is named, and how a variable is kept apart from
#: another of the same name elsewhere. Neither name can be a predicate's, so neither can collide with one.
INVENTED = "#invented{number}"
APART = "{name}#{number}"


class Clausifier:
    """A formula as the clauses that say the same thing.

    Nobody states a rule as a disjunction of literals, and the engine reasons with nothing else. So a rule is
    written the way it is meant — if this then that, for every thing, there is some thing such that — and this does
    the mechanical work of turning it into the one form, once, so that neither side has to bend. It is also what
    lets a rule decoded out of a sentence arrive in whatever shape the sentence had.

    Four steps, in this order because each one needs the last:

    1. **Arrows out.** `a → b` becomes `not a or b`, and `a ↔ b` becomes both directions, so only and, or and not
       are left.
    2. **Denials pushed in**, until nothing is denied but a single thing said. Denying "for every" gives "there is
       some", and denying "there is some" gives "for every", which is why this has to happen before the quantifiers
       are dealt with.
    3. **The things conjured up get names.** "There is some thing such that" is replaced by a thing invented for
       the purpose, and where it sits inside "for every", the invented thing is a function of those — because which
       thing there is may depend on which thing you started with. This is the step that cannot be undone, and the
       one that makes the result say the same thing without meaning the same thing.
    4. **Ors distributed over ands**, which leaves a conjunction of disjunctions, and each disjunction is a clause.

    Variables are renamed apart first, so a formula that uses the same letter twice for two different things does
    not come out saying they were one.

    It keeps nothing but a counter for the names it invents, reset at each call, so the same formula clausified
    twice gives the same clauses."""

    def clauses(self, formula: Formula, name: str = "", probability: float = 1.0) -> tuple[Clause, ...]:
        """The clauses saying what that formula says.

        Each carries the name and the probability given, so a rule that was stated once can be recognised in its
        pieces and weighed as one thing."""
        invented = itertools.count(1)
        apart = self._apart(formula, itertools.count(1), Substitution())
        plain = self._push(self._arrows(apart), False)
        named = self._name(plain, invented, ())
        found = tuple(
            Clause(literals, probability, name)
            for literals in self._distribute(named)
            if not self._trivial(literals)
        )
        logger.debug("Clausified %s into %d clauses", name or "a formula", len(found))
        return found

    def _arrows(self, formula: Formula) -> Formula:
        """Implications and equivalences written with and, or and not."""
        if isinstance(formula, Implies):
            return Or((Not(self._arrows(formula.premise)), self._arrows(formula.conclusion)))
        if isinstance(formula, Iff):
            left, right = self._arrows(formula.left), self._arrows(formula.right)
            return And((Or((Not(left), right)), Or((Not(right), left))))
        if isinstance(formula, And):
            return And(tuple(self._arrows(one) for one in formula.parts))
        if isinstance(formula, Or):
            return Or(tuple(self._arrows(one) for one in formula.parts))
        if isinstance(formula, Not):
            return Not(self._arrows(formula.body))
        if isinstance(formula, ForAll):
            return ForAll(formula.variables, self._arrows(formula.body))
        if isinstance(formula, Exists):
            return Exists(formula.variables, self._arrows(formula.body))
        return formula

    def _push(self, formula: Formula, denied: bool) -> Formula:
        """The formula with every denial pushed down onto the things said."""
        if isinstance(formula, Not):
            return self._push(formula.body, not denied)
        if isinstance(formula, Atom):
            literal = formula.literal
            return Atom(literal.denied if denied else literal)
        if isinstance(formula, Truth):
            return Truth(not formula.value if denied else formula.value)
        if isinstance(formula, And):
            parts = tuple(self._push(one, denied) for one in formula.parts)
            return Or(parts) if denied else And(parts)
        if isinstance(formula, Or):
            parts = tuple(self._push(one, denied) for one in formula.parts)
            return And(parts) if denied else Or(parts)
        if isinstance(formula, ForAll):
            body = self._push(formula.body, denied)
            return Exists(formula.variables, body) if denied else ForAll(formula.variables, body)
        if isinstance(formula, Exists):
            body = self._push(formula.body, denied)
            return ForAll(formula.variables, body) if denied else Exists(formula.variables, body)
        return formula

    def _name(self, formula: Formula, invented: "itertools.count[int]", under: tuple[Variable, ...]) -> Formula:
        """Every "there is some" replaced by a thing invented for it, a function of the variables it sits under."""
        if isinstance(formula, ForAll):
            return self._name(formula.body, invented, (*under, *formula.variables))
        if isinstance(formula, Exists):
            agreed = Substitution()
            for variable in formula.variables:
                name = INVENTED.format(number=next(invented))
                agreed = agreed.bound(variable, Functor(name, under) if under else Functor(name, ()))
            return self._name(self._substituted(formula.body, agreed), invented, under)
        if isinstance(formula, And):
            return And(tuple(self._name(one, invented, under) for one in formula.parts))
        if isinstance(formula, Or):
            return Or(tuple(self._name(one, invented, under) for one in formula.parts))
        return formula

    def _distribute(self, formula: Formula) -> tuple[tuple[Literal, ...], ...]:
        """The formula as a conjunction of disjunctions, each disjunction the literals of one clause."""
        if isinstance(formula, Atom):
            return ((formula.literal,),)
        if isinstance(formula, Truth):
            return () if formula.value else ((),)
        if isinstance(formula, And):
            return tuple(one for part in formula.parts for one in self._distribute(part))
        if isinstance(formula, Or):
            found: tuple[tuple[Literal, ...], ...] = ((),)
            for part in formula.parts:
                found = tuple(
                    self._merged(mine, theirs) for mine in found for theirs in self._distribute(part)
                )
            return found
        raise ValueError(f"A {type(formula).__name__} is left where only literals, and, or should be")

    def _merged(self, one: tuple[Literal, ...], other: tuple[Literal, ...]) -> tuple[Literal, ...]:
        """The two sets of literals together, each literal once."""
        found = list(one)
        found.extend(literal for literal in other if literal not in found)
        return tuple(found)

    def _trivial(self, literals: tuple[Literal, ...]) -> bool:
        """Whether the clause says nothing: something and its own denial are both in it, so it is true whatever."""
        return any(one.denied in literals for one in literals)

    def _apart(self, formula: Formula, numbers: "itertools.count[int]", agreed: Substitution) -> Formula:
        """Every quantifier's variables renamed to something nothing else uses."""
        if isinstance(formula, (ForAll, Exists)):
            renamed = agreed
            variables = []
            for variable in formula.variables:
                fresh = Variable(APART.format(name=variable.name, number=next(numbers)), variable.sort)
                renamed = renamed.bound(variable, fresh)
                variables.append(fresh)
            body = self._apart(formula.body, numbers, renamed)
            return ForAll(tuple(variables), body) if isinstance(formula, ForAll) else Exists(tuple(variables), body)
        if isinstance(formula, Atom):
            return Atom(agreed.applied(formula.literal))
        if isinstance(formula, Not):
            return Not(self._apart(formula.body, numbers, agreed))
        if isinstance(formula, And):
            return And(tuple(self._apart(one, numbers, agreed) for one in formula.parts))
        if isinstance(formula, Or):
            return Or(tuple(self._apart(one, numbers, agreed) for one in formula.parts))
        if isinstance(formula, Implies):
            return Implies(self._apart(formula.premise, numbers, agreed), self._apart(formula.conclusion, numbers, agreed))
        if isinstance(formula, Iff):
            return Iff(self._apart(formula.left, numbers, agreed), self._apart(formula.right, numbers, agreed))
        return formula

    def _substituted(self, formula: Formula, agreed: Substitution) -> Formula:
        if isinstance(formula, Atom):
            return Atom(agreed.applied(formula.literal))
        if isinstance(formula, Not):
            return Not(self._substituted(formula.body, agreed))
        if isinstance(formula, And):
            return And(tuple(self._substituted(one, agreed) for one in formula.parts))
        if isinstance(formula, Or):
            return Or(tuple(self._substituted(one, agreed) for one in formula.parts))
        if isinstance(formula, ForAll):
            return ForAll(formula.variables, self._substituted(formula.body, agreed))
        if isinstance(formula, Exists):
            return Exists(formula.variables, self._substituted(formula.body, agreed))
        return formula

    def _term(self, term: Term, agreed: Substitution) -> Term:
        return agreed.applied(term)
