import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from math import comb

from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.example import Example
from openmind.inference.model.pursuit import Pursuit
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.inference.service.hypothesis_table import HypothesisTable
from openmind.inference.service.hypothesis_tester import HypothesisTester
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)


class Sniper:
    """What refuses one candidate the rules wrongly allow, searched for as long as it takes.

    **Every error a mature set of constraints makes is the same error.** Measured on a run's own forty-eight
    constraints against a position twelve moves in: fourteen thousand four hundred candidates, a hundred and
    eighteen left standing, and all forty-three legal moves among them. Not one constraint refuses a legal
    move. So there is nothing to mend and everything to find, and what is missing is concentrated — thirty-one
    of the seventy-five wrongly allowed are a rook moving in a way no rook moves, which takes three conditions
    to say.

    **The ordinary learner cannot spend its time there.** It is given a position's unaccounted cases and sixty
    seconds between them, so each gets a shallow look, and a shallow look is right for the easy refusals that
    dominate early. Run deeper as a whole arm for twelve positions it did not pay, for exactly that reason. On
    the residue it should, and nothing was aiming at the residue.

    **Ranked by how much else it fixes.** A body refusing this candidate and forty others spread over ten
    boards is a rule about the game; one refusing forty on the same board may be a rule about that board. So
    the candidates the rules wrongly allow are pooled across positions and a body is scored against the pool.

    It keeps nothing: built once, it is given the pursuit, the guard, the table and the pool on every call."""

    def __init__(
        self,
        refusal_learner: RefusalLearner | None = None,
        clock: Callable[[], float] = time.monotonic,
        between_clocks: int = 64,
    ) -> None:
        self._learner = RefusalLearner() if refusal_learner is None else refusal_learner
        self._tester = HypothesisTester(self._learner)
        self._clock = clock
        # How many bodies are tried between one look at the clock and the next.
        #
        # Not a limit on anything: the search runs until its time is up or it is finished, and this only says
        # how coarsely it notices. It is here rather than inside the testing because a range that stopped
        # part-way would leave nowhere to resume from — the pursuit would either skip what was never tried or
        # try again what was. A whole range always finishes, so where it got to is exactly where it ends.
        self._between_clocks = between_clocks

    def started(
        self,
        case: Example,
        readings: CandidateReadings | None = None,
        among: Sequence[Clause] = (),
    ) -> Pursuit:
        """A pursuit of that candidate, its readings tied once and kept, including what could happen after it.

        Tied by the readings the case was read with, where the caller has them: what a rule may be built from
        is the couple of dozen readings the candidate is *in*, not the sixty-four the position carries about
        squares the candidate never names.

        **And this is where reading the board a candidate leads to becomes affordable.** A later reading costs
        a ply of lookahead, which over a position's fourteen thousand candidates is why the question was asked
        lazily and answered opaquely. A pursuit is one candidate; a pool is scores of them. A ply each is
        nothing, and what it buys is that the rules about what could happen next are in the space the search
        combines rather than outside it.

        `among` is what is believed so far, and it is not optional. The question of whether the other side
        could reply is put to the rules below, and with none the answer is yes for every move there is —
        which is how king safety once refused all twenty legal moves of the opening position."""
        tying = CandidateReadings() if readings is None else readings
        asking = self._learner.hypothetical
        later = () if asking is None else asking.happenings(
            case, lambda one: any(self._learner.covers(clause, one, among) for clause in among)
        )
        told = Example((*case.literals, *later), case.holds, case.where) if later else case
        return Pursuit(told, tying.tied(told.literals))

    def pursue(
        self,
        pursuit: Pursuit,
        guard: CaseIndex,
        table: HypothesisTable,
        pool: Sequence[Example] = (),
        seconds: float = 300.0,
    ) -> Pursuit:
        """That pursuit carried on for that long, and where it got to.

        It resumes at the size and place the pursuit carries, prunes every size past the first by what slipped
        one condition shorter, and writes everything it tries into the table — what lost as well as what won,
        since what loses here is what the next pursuit finds already answered.

        It comes back whether or not it found anything. Where a size finishes with nothing having slipped in
        it, the pursuit comes back exhausted: no longer body can beat what stands, so there is nothing further
        to do and coming back again would find the same."""
        if pursuit.exhausted:
            return pursuit
        deadline, began = self._clock() + seconds, self._clock()
        size, at, exhausted = max(pursuit.size, 1), pursuit.at, False
        while self._clock() < deadline:
            whole = comb(len(pursuit.offered), size)
            if at >= whole:
                if not table.slipping(size):
                    exhausted = True
                    break
                size, at = size + 1, 0
                continue
            stop = min(at + self._between_clocks, whole)
            table.tell(
                self._tester.tried(
                    pursuit.offered, size, pool, guard, start=at, stop=stop,
                    beyond=table.slipping(size - 1) if size > 1 else None,
                )
            )
            at = stop
        standing = [one.clause for one in table.useful() if self._learner.covers(one.clause, pursuit.case)]
        ranked = self.ranked(standing, pool)
        carried = replace(
            pursuit,
            size=size,
            at=at,
            found=ranked[0][0] if ranked else None,
            seconds=pursuit.seconds + (self._clock() - began),
            exhausted=exhausted,
        )
        logger.info(
            "Pursued a candidate the rules allow and the game refuses: %s%s",
            carried.readable,
            f", refusing {ranked[0][1]} others" if ranked else "",
        )
        if carried.exhausted and carried.found is None:
            logger.warning(
                "Nothing the readings can say tells this candidate from the legal moves, at any length: %s. "
                "That is a reading the vocabulary has not got, not a search that ran out of time",
                dict(sorted((one.predicate, str(one.arguments)) for one in pursuit.case.literals)[:4]),
            )
        return carried

    def ranked(self, standing: Sequence[Clause], pool: Sequence[Example]) -> tuple[tuple[Clause, int], ...]:
        """Each body with how many of the pool it also refuses, most first, ties broken by what it costs.

        The pool is the candidates the rules allow and the game refuses, gathered across positions. Scoring
        against it rather than against one board is what tells a rule about the game from a rule about a
        board — two bodies refusing this candidate are told apart by everything else they refuse."""
        scored = [(one, sum(1 for case in pool if self._learner.covers(one, case))) for one in standing]
        return tuple(sorted(scored, key=lambda held: (-held[1], self._learner.cost(held[0]), held[0].readable)))
