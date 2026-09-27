import logging
from collections.abc import Mapping, Sequence

from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.misprediction import Misprediction
from openmind.inference.model.surprise import Surprise
from openmind.inference.service.chance_fitter import ChanceFitter
from openmind.inference.service.clause_learner import ClauseLearner
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal

logger = logging.getLogger(__name__)


class ClauseChallenger:
    """What would break these clauses, and what happened when something did.

    **A learner that only ever sees what a game happens to offer learns what the game happens to offer.** Waiting
    for a counter-example means waiting on however often the game deals one, and the rules that almost never come
    up are exactly the ones worth having and exactly the ones that never arrive. So instead each clause is asked
    what would break it: for every literal it leans on, find the case that meets all the others and fails this one.
    That is a case to go and build, not a case to wait for.

    **How long a clause stood decides what breaking it means.** Broken by the fifth case it met, it was a guess and
    it is simply wrong. Broken by the three-thousandth, it is a rule of the game that almost never comes up, and
    meeting it is the most informative thing that has happened — so the case is kept, and every clause learned
    afterwards is made to face it rather than wait for chance to bring it round again.

    **What is not accounted for is reported rather than hidden.** Cases the game refuses that no clause refuses,
    cases nobody knew were possible, and outcomes that came out otherwise than expected. Each says a different
    thing about where to look, and none of them is a failure of the learning — they are the learning's output just
    as much as the clauses are.

    It keeps nothing: built once, it is given the clauses and the cases on every call."""

    def __init__(self, clause_learner: ClauseLearner | None = None, chance_fitter: ChanceFitter | None = None) -> None:
        self._learner = ClauseLearner() if clause_learner is None else clause_learner
        self._chances = ChanceFitter() if chance_fitter is None else chance_fitter

    def challenges(self, clauses: Sequence[Clause]) -> tuple[tuple[Clause, Literal], ...]:
        """Each clause paired with each literal it leans on: what to look for an exception to.

        A literal is only worth what it turns away. One that turns away nothing the game allows is a reason; one
        that turns away something is a clause stated too narrowly, and the case it turned away is the edge case
        that says so. What each pair asks for is a case meeting every other literal of its clause and failing this
        one."""
        return tuple((clause, literal) for clause in clauses for literal in clause.body)

    def asked(self, clause: Clause, without: Literal) -> Clause:
        """The challenge as something to go and satisfy: that clause with one literal left out.

        A case covered by this and not by the clause itself is the exception being looked for."""
        return Clause(tuple(one for one in clause.literals if one != without.denied), clause.probability, clause.name)

    def challenged(
        self, clauses: Sequence[Clause], examples: Sequence[Example]
    ) -> dict[tuple[Literal, bool], int]:
        """What each literal of each clause costs, counted over the cases it was asked about.

        Against it, `(literal, False)`: the cases that hold, meet every other literal of its clause, and fail this
        one. Each is a way of holding that the clause cannot state, and where there are any, the literal is wrong
        as it stands — it is either to be dropped or to be split into the cases it was standing in for.

        For it, `(literal, True)`: the cases that do not hold and its clause covers with that literal held. Those
        are what the literal failed to turn away, and they say the clause is loose somewhere else."""
        counted: dict[tuple[Literal, bool], int] = {}
        for clause, literal in self.challenges(clauses):
            widened = self.asked(clause, literal)
            for example in examples:
                if example.holds and self._learner.covers(widened, example) and not self._learner.covers(clause, example):
                    counted[(literal, False)] = counted.get((literal, False), 0) + 1
                if not example.holds and self._learner.covers(clause, example):
                    counted[(literal, True)] = counted.get((literal, True), 0) + 1
        return counted

    def refuted(
        self, clauses: Sequence[Clause], examples: Sequence[Example], stood: Mapping[str, int] | None = None
    ) -> tuple[tuple[Clause, ...], tuple[Surprise, ...]]:
        """The clauses that survived those cases, and what broke the rest.

        A clause is broken by a case it covers that does not hold. How long it had stood comes from the caller,
        which is what knows how much has been looked at."""
        standing = dict(stood or {})
        surviving: list[Clause] = []
        broken: list[Surprise] = []
        for clause in clauses:
            against = next((one for one in examples if not one.holds and self._learner.covers(clause, one)), None)
            if against is None:
                surviving.append(clause)
                continue
            held = standing.get(clause.readable, 0)
            broken.append(Surprise(clause.readable, held, None, None, (), against))
            logger.info("%s broke after standing %d", clause.readable, held)
        return tuple(surviving), tuple(broken)

    def unexplained(self, clauses: Sequence[Clause], examples: Sequence[Example]) -> tuple[Example, ...]:
        """The cases that do not hold and no clause rules out.

        Either a reading is missing, or what rules them out is not about what is being read at all. Saying so is
        the point: a learner that quietly allowed them would be claiming to have accounted for the game."""
        found = tuple(
            one for one in examples if not one.holds and not self._learner.covered(clauses, one)
        )
        refused = sum(1 for one in examples if not one.holds)
        if found:
            logger.warning(
                "%d of %d cases that don't hold are ones no clause rules out: either a reading is missing or what "
                "rules them out isn't about what is being read",
                len(found), refused,
            )
        return found

    def unforeseen(self, considered: Sequence[Literal], held: Sequence[Literal]) -> tuple[Literal, ...]:
        """What turned out to be so and was never among the candidates: what OMF did not know could be done.

        A clause can be wrong two ways. It can allow what the game refuses, which a candidate case refutes; or the
        game can allow something that was never offered as a candidate at all, which nothing refutes because
        nothing ever asked. The second is invisible to any amount of checking against candidates."""
        offered = set(considered)
        found = tuple(one for one in held if one not in offered)
        if found:
            logger.info("%d things turned out possible that were never among the %d considered", len(found), len(offered))
        return found

    def mispredicted(
        self, clauses: Sequence[Clause], seen: Sequence[Example], budget: InferenceBudget
    ) -> tuple[Misprediction, ...]:
        """Where what happened and what was expected to happen part company.

        Told apart into the two kinds that want different repairs: an outcome nothing predicted is a clause
        missing, and one predicted at the wrong rate is a count to correct. Run together, a game with chance in it
        looks like a game whose rules keep breaking, and the model is torn up every time a coin lands the other
        way."""
        found: list[Misprediction] = []
        for outcome in self._outcomes(seen):
            about = [one for one in seen if self._says(one, outcome)]
            held = self._chances.counted(sum(1 for one in about if one.holds), len(about))
            covering = [one for one in clauses if any(self._learner.covers(one, case) for case in about)]
            if not covering:
                found.append(Misprediction(outcome, None, held, True, about[0].where if about else None))
                continue
            expected = max((one.probability for one in covering), default=1.0)
            if abs(expected - held.value) > held.spread:
                found.append(Misprediction(outcome, self._chances.counted(0, 0), held, False, about[0].where if about else None))
        unforeseen = sum(1 for one in found if one.unforeseen)
        if found:
            logger.info(
                "%d outcomes came out otherwise than expected: %d nothing predicted, %d at the wrong rate",
                len(found), unforeseen, len(found) - unforeseen,
            )
        return tuple(found)

    def _outcomes(self, seen: Sequence[Example]) -> tuple[Literal, ...]:
        """Each distinct thing the cases were about."""
        found: dict[Literal, None] = {}
        for example in seen:
            for literal in example.literals:
                found.setdefault(Literal(literal.predicate, ()))
        return tuple(found)

    def _says(self, example: Example, outcome: Literal) -> bool:
        return bool(example.says(outcome.predicate))
