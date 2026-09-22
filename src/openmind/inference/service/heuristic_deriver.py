import logging
from collections.abc import Sequence

from openmind.inference.model.derivation import Derivation
from openmind.inference.model.example import Example
from openmind.inference.model.derivation_step import GIVEN, RESOLVED, DerivationStep
from openmind.inference.model.heuristic import OPTIMAL, Heuristic
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.parameter import Parameter
from openmind.inference.model.signature import Signature
from openmind.inference.service.clause_learner import ClauseLearner
from openmind.inference.service.subsumer import Subsumer
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Variable
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: What a derived value is called, and what it is about.
AFFORDS = "what {about} affords"
WORTH = "worth of {about}"

#: How many combinations of readings are tried before the count is taken as read.
ENOUGH = 200_000


class HeuristicDeriver:
    """What the rules imply about value, said without being asked.

    It is not a question-answering service. Nothing is handed it a list of the things in a game with an
    instruction to price them — it would have to be told what the things are, and being told is the one thing the
    bootstrap cannot rely on. It reads its own rules, finds what they are about, and comes back with what those
    things are worth. The caller learns there are knights from the answer, not before it.

    **How a value is arrived at.** A rule says what a thing may do. Counting the ways it may do it is counting the
    combinations of readings the rule admits — a rule asking for two steps, neither straight nor diagonal, admits
    few; a rule asking only that the way be clear admits many. Nothing is played and no board is needed: the
    readings and the values they were seen taking are enough, so a game with no board yields to the same counting
    as a game with one.

    **And the ordering comes free.** Where everything one thing's rules admit another's admit too, the second
    affords at least what the first does, in every position, always. That is subsumption, and it needs no counting
    at all.

    What comes back is starting points, not answers. A thing whose rules admit fourteen ways is worth looking for
    near fourteen; what it is really worth is what the games say.

    It keeps nothing: built once, it is given the rules on every call."""

    def __init__(self, subsumer: Subsumer | None = None, clause_learner: ClauseLearner | None = None) -> None:
        self._subsumer = Subsumer() if subsumer is None else subsumer
        self._learner = ClauseLearner() if clause_learner is None else clause_learner

    def derive(
        self,
        clauses: Sequence[Clause],
        cases: Sequence[Example],
        budget: InferenceBudget,
        signature: Signature | None = None,
    ) -> tuple[Heuristic, ...]:
        """Everything the rules say about what things are worth.

        Nothing is asked for and nothing is named. What comes back is one heuristic per thing the rules turn out
        to be about, each with the value its own rules admit and the reasoning that reached it."""
        about = self._about(clauses)
        found: list[Heuristic] = []
        for thing, mine in sorted(about.items(), key=lambda held: str(held[0])):
            admitted = sum(1 for one in cases if any(self._permits(clause, one) for clause in mine))
            found.append(
                Heuristic(
                    thing,
                    mine[0],
                    OPTIMAL,
                    self._why(thing, mine, admitted),
                    parameters=(Parameter(WORTH.format(about=thing), float(admitted), AFFORDS.format(about=thing)),),
                )
            )
        logger.info(
            "The rules are about %d things, and say what each affords: %s",
            len(found),
            "; ".join(one.readable for one in found),
        )
        return tuple(found)

    def ordered(self, clauses: Sequence[Clause]) -> tuple[tuple[Value, Value], ...]:
        """Which things afford at least what another does, from the rules alone and no counting.

        Where everything one thing's rules allow another's allow too, the second affords at least what the first
        does — in every position, always, and before anything is looked at."""
        about = self._about(clauses)
        found = self._subsumer.ordered(
            {thing: tuple(self._besides(one, thing) for one in mine) for thing, mine in about.items()}
        )
        for wider, narrower in found:
            logger.info("Whatever %s may do, %s may do: it affords at least as much", narrower, wider)
        return found

    def _besides(self, clause: Clause, thing: Value) -> Clause:
        """That rule with the reading naming the thing set aside.

        What a thing affords and when it may do it are different questions, and the reading saying which thing is
        standing there answers the second. Left in, it makes every thing incomparable with every other — each
        rule asks for its own thing and no other rule asks for that — so the ordering that ought to fall out of
        the rules falls out of nothing. Set aside, what is left is what the thing may do, which is the question."""
        return Clause(
            tuple(
                one
                for one in clause.literals
                if not (one.arguments and one.arguments[-1] == Constant(thing))
            ),
            clause.probability,
            clause.name,
        )

    def _about(self, clauses: Sequence[Clause]) -> dict[Value, list[Clause]]:
        """What the rules turn out to be about, and which rules are about each.

        A rule is about whatever it pins down and does not vary: the kinds named in it that the reading's value
        place holds outright. Nothing is declared and nothing is looked up — a game whose rules pin down no kind
        simply has one thing, which is the game."""
        found: dict[Value, list[Clause]] = {}
        for clause in clauses:
            for thing in self._named(clause):
                found.setdefault(thing, []).append(clause)
        return found

    def _named(self, clause: Clause) -> tuple[Value, ...]:
        """The kinds a rule pins down: the values it asks for outright, where other rules vary them.

        The value place of a reading is its last, since a reading names what it is about and then what was read of
        it. A rule asking for that place to hold one particular thing is a rule about that thing."""
        found: dict[Value, None] = {}
        for literal in clause.body:
            if not literal.arguments:
                continue
            last = literal.arguments[-1]
            if isinstance(last, Constant) and isinstance(last.name, str) and last.name:
                found.setdefault(last.name)
        return tuple(found)

    def _admits(self, clause: Clause, signature: Signature, cases: Sequence[Example]) -> int:
        """How many different things the rule permits, counted over what an action can actually be.

        **Not a product of how many values each reading takes.** A rule speaks of many readings — how many rows
        apart, how many columns apart, how many steps, whether on a diagonal — and they are not separate choices.
        They are several readings of one and the same thing, so fixing that thing fixes every one of them.
        Multiplying their counts together counts combinations that cannot occur: a rule admitting thirteen ways of
        moving comes out in the hundred thousands, and the number is not wrong by a factor, it is not a count of
        anything.

        So the counting is done over the one joint thing the readings are of: each case the rule was learned from
        is one way of acting, and what the rule admits is how many distinct ones it covers. A rule pinning
        everything down admits one; a rule leaving the way open admits as many as there are ways. That is a count
        of things that exist, which is what makes it a deduction rather than an arithmetic coincidence."""
        return sum(1 for one in cases if self._permits(clause, one))

    def _permits(self, clause: Clause, case: Example) -> bool:
        """Whether the rule allows that way of acting, its readings taken together."""
        return self._learner.covers(clause, case)

    def _why(self, thing: Value, clauses: Sequence[Clause], admitted: int) -> Derivation:
        """The reasoning that reached it: the rules it read, and the counting it did over them."""
        steps = tuple(
            DerivationStep(number + 1, GIVEN, (), clause) for number, clause in enumerate(clauses)
        )
        reached = Clause(
            (Literal(WORTH.format(about=thing), (Constant(thing), Constant(admitted))),),
            1.0,
            AFFORDS.format(about=thing),
        )
        return Derivation(
            reached,
            (*steps, DerivationStep(len(steps) + 1, RESOLVED, tuple(range(1, len(steps) + 1)), reached)),
        )
