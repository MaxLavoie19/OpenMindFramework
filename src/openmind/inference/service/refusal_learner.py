import logging
import time
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

from openmind.inference.constant.refusal_constant import REFUSED
from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.disagreement import Disagreement
from openmind.inference.model.evidence import Evidence
from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.substitution import Substitution
from openmind.inference.service.candidate_readings import VALUING, CandidateReadings
from openmind.inference.service.evaluable_predicates import EvaluablePredicates
from openmind.inference.service.hypothesis_table import HypothesisTable
from openmind.inference.service.hypothesis_tester import HypothesisTester
from openmind.inference.service.hypothetical import ALLOWED, ALLOWED_AFTER, TAKEN_AFTER, Hypothetical
from openmind.inference.service.unifier import Unifier
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Term, Variable
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: How a variable that generalising introduces is named.
FRESH = "X{number}"


class RefusalLearner:
    """The constraints that account for what a game refuses.

    **There are no generators here.** A constraint covers candidates the game refuses, and a candidate is legal
    exactly when no constraint covers it. Nothing produces the legal moves; they are what is left.

    **Refusing is not one thing.** A way may be blocked, a thing may not be the mover's, a king may be left where
    it can be taken. No single set of conditions holds of every refused candidate, which is why asking what they
    all have in common gives nothing. So one constraint is grown to cover some of them, what it covers is set
    aside, and another is grown for the rest.

    **Each constraint is grown from a case, upward.** One case is said outright — everything read of it — and the
    clause is then widened to take in further cases: where two agree a term stays, where they differ a variable
    goes in, and the same differing pair always gives the same variable. A widening that would cover a candidate
    the game *allows* is refused outright and another case is tried, because a constraint that turns away a legal
    move costs a move OMF will never make.

    So a constraint says less the more cases it has met, and what it ends up saying is what those cases had in
    common — never something proposed and tested. The variables come out by construction rather than by search,
    which is the whole reason for working this way round: a case whose mover is one side and a case whose mover is
    the other give the *same* variable wherever either side appeared, so one clause about whoever is acting falls
    out of widening two cases.

    **The work is where the cases are few.** A position offers thousands of refused candidates and a handful of
    legal ones, so checking that a constraint refuses nothing it should not is a handful of questions, while
    covering what it should is where the widening goes. That asymmetry is why this direction is the affordable
    one.

    It keeps nothing: built once, it is given the cases on every call."""

    def __init__(
        self,
        readings: CandidateReadings | None = None,
        unifier: Unifier | None = None,
        evaluable: EvaluablePredicates | None = None,
        hypothetical: Hypothetical | None = None,
        clock: Callable[[], float] = time.monotonic,
        further: int = 1,
        per_clause: float = 1.0,
        exponent: float = 2.0,
    ) -> None:
        self._readings = CandidateReadings() if readings is None else readings
        self._unifier = Unifier() if unifier is None else unifier
        self._evaluable = EvaluablePredicates() if evaluable is None else evaluable
        self._hypothetical = hypothetical
        self._clock = clock
        self._tester = HypothesisTester(self)
        # How many sizes past the first that works to keep looking, and what a constraint costs to say.
        #
        # **A longer body can refuse more, and stopping at the first size that works never finds out.** Adding
        # a condition only narrows, so among bodies built from the same readings a longer one always refuses
        # less — but a longer body whose every shorter part turns away a legal move is reached by no shorter
        # search, and it may refuse a great deal. Measured over eight cases a run's constraints still left
        # standing, against a guard of eighteen hundred legal moves: three stopped at one condition and took a
        # body refusing five to six hundred where two conditions would have refused nine hundred and sixty, and
        # a fourth gained again at three. Half the cases lost between a third and a half of what was there.
        #
        # The price is the one the distiller already puts on a constraint — what having it at all costs, plus
        # its length raised to a power — so that the search and the distilling agree about what a condition is
        # worth. Without the price, widening the window would simply take the longest body every time.
        self._further = further
        self._per_clause = per_clause
        self._exponent = exponent

    @property
    def clock(self) -> Callable[[], float]:
        """What it tells the time by, so that anything sharing its budget tells the time the same way."""
        return self._clock

    def cost(self, clause: Clause) -> float:
        """What that constraint costs to say: having it at all, plus its length raised to a power.

        The same price the distiller puts on one, so that what the search prefers and what the distilling keeps
        are the same preference asked at two moments. A search choosing by coverage alone takes the longest body
        it is offered, since a longer body is reached only where every shorter part of it turns away a legal
        move — and those are exactly the bodies that refuse most."""
        return self._per_clause + float(len(clause.body)) ** self._exponent

    @property
    def hypothetical(self) -> Hypothetical | None:
        """How it asks about actions nobody is taking, where it can ask at all.

        Said aloud so a caller can tell which of a set of rules sit above the layer — that is a fact about what
        the rules refer to, worked out from them and declared nowhere, and a caller putting each rule to every
        candidate on its own needs it to know which ones it cannot afford to ask."""
        return self._hypothetical

    def learn(
        self,
        examples: Sequence[Example],
        budget: InferenceBudget,
        starting: Sequence[Clause] = (),
        least: int = 1,
        positions: int = 1,
        coverage: Mapping[Clause, frozenset[int]] | None = None,
        guard: Sequence[Example] = (),
        table: HypothesisTable | None = None,
    ) -> tuple[Clause, ...]:
        """The constraints accounting for the cases the game refuses, each covering at least `least` of them.

        There is no limit on how many. How many a game needs is a fact about the game and not a number to guess
        beforehand, and a limit guessed too low truncates the game: the constraints that would have refused a
        whole kind of move are never reached, and what is missing looks like something the learner could not
        learn.

        `starting` is what is already believed. A case already accounted for is left alone, and a case that is not
        is first tried as a widening of something already held before anything new is begun. That is what lets an
        agent learn position by position rather than from a corpus: each position asks only what it did not
        already know, and what it knew is widened rather than replaced.

        `guard` is every move the game has allowed elsewhere, added to this position's own for the purpose of
        refusing a candidate constraint. **Nothing in one position's twenty legal moves tells a rule from a
        postcode**: `refused :- self(1, 1)` turns away nothing legal in a position where that square holds an
        enemy piece, and turns away everything in the next. A constraint is only shown to be wrong by evidence it
        was not grown against, so the guard is what makes the difference sayable at all. The distiller has been
        given this all along — its own note says a fold is "checked against all of them and not only against the
        position at hand" — and the learner has been judging against twenty."""
        deadline = self._clock() + budget.seconds
        refused = [one for one in examples if one.holds]
        allowed = [one for one in examples if not one.holds]
        # Deduped, because a caller that accumulates the guard has this position's legal moves in it already and
        # should not have to reason about that. A case read identically twice guards nothing twice.
        index = CaseIndex(tuple(dict.fromkeys((*allowed, *guard))))
        found: list[Clause] = list(starting)
        held = self.coverage(found, examples) if coverage is None else coverage
        accounted = frozenset().union(*held.values(), frozenset())
        left = [one for number, one in enumerate(examples) if one.holds and number not in accounted]
        given, stopped = len(found), "every refusal is accounted for"
        while left:
            if self._clock() >= deadline:
                stopped = "the time ran out"
                break
            grown, used, replaced = self._widened(left, index, deadline, found, positions, table, examples)
            if grown is None:
                stopped = "nothing further could be generalised"
                break
            covered = {id(one) for one in left if self.covers(grown, one)}
            if len(covered) < least:
                stopped = f"the next constraint would account for fewer than {least}"
                break
            found = [one for one in found if one is not replaced]
            found.append(grown)
            left = [one for one in left if id(one) not in covered]
            logger.debug(
                "Kept %s, grown from %d cases, refusing %d more", grown.readable, len(used), len(covered)
            )
        wrongly = sum(1 for one in allowed if self.refuses(found, one))
        logger.info(
            "Learned %d constraints (%d given) refusing %d of %d refused candidates, wrongly refusing %d of %d "
            "allowed: %s",
            len(found), given, len(refused) - len(left), len(refused), wrongly, len(allowed), stopped,
        )
        return tuple(found)

    def computes(self, predicate: str) -> bool:
        """Whether that reading is answered by computing rather than by a case having it."""
        return self._evaluable.evaluable(predicate) or predicate in (ALLOWED, ALLOWED_AFTER, TAKEN_AFTER)

    def coverage(
        self, clauses: Sequence[Clause], examples: Sequence[Example]
    ) -> dict[Clause, frozenset[int]]:
        """Which of those cases each constraint covers, by their place in the list.

        Worked out once and handed to whoever needs it. Learning asks it to find what is still unaccounted for
        and distilling asks it to know what may be given up, and each computing it for itself means asking every
        constraint about every candidate twice — which at a hundred and eighty constraints and four thousand
        candidates is most of a position's budget spent arriving back where the other one started."""
        return {
            clause: frozenset(
                number for number, one in enumerate(examples) if self.covers(clause, one, clauses)
            )
            for clause in clauses
        }

    def refuses(self, clauses: Sequence[Clause], example: Example) -> bool:
        """Whether any of them refuses that candidate.

        There is no second list to consult and no exception to check. This is the whole of legality."""
        return any(self.covers(one, example, clauses) for one in clauses)

    def scored(
        self,
        clauses: Sequence[Clause],
        evidence: Evidence,
        domains: Mapping[str, Sequence[Value]],
        cases: Sequence[Example] | None = None,
    ) -> Disagreement:
        """How those constraints stand against a position, candidate by candidate.

        `cases` is that position's candidates already read, where the caller has them. Reading fourteen thousand
        candidates is a third of what a position costs, and a caller that learned from them is holding the very
        same list — read from the same position under the same domains, in the order `candidates` gives. Left
        out, they are read here, which is what a caller asking about constraints it did not learn from wants."""
        if cases is None:
            cases = self._readings.cases(evidence, domains)
        allowed, forbade = [], []
        for action, case in zip(self._readings.candidates(evidence, domains), cases, strict=True):
            refusing = self.refuses(clauses, case)
            if case.holds and not refusing:
                allowed.append(action)
            elif not case.holds and refusing:
                forbade.append(action)
        found = Disagreement(evidence.where, tuple(allowed), tuple(forbade))
        logger.info("Against one position, %d constraints leave %s", len(clauses), found.readable)
        return found

    def wrongly(self, clauses: Sequence[Clause], evidence: Evidence) -> int:
        """How many of the actions the game allows there those constraints refuse.

        **The mistake nothing else will ever tell it about.** A candidate wrongly let through turns up the moment
        the game lists a move the constraints thought impossible; a legal move wrongly refused is a move OMF will
        never make, never propose, and never hear was available. So a position holding one is evidence that
        exists nowhere else, and finding such positions is worth looking for them.

        Cheap on purpose. It reads the moves the game listed and not the whole candidate space, which is what
        lets a position be judged in the time it takes to read forty cases rather than fourteen thousand."""
        cases = self._readings.of_allowed(evidence)
        return sum(1 for one in cases if self.refuses(clauses, one))

    def repaired(
        self, clause: Clause, keeping: Sequence[Example], without: Sequence[Example]
    ) -> Clause | None:
        """That constraint with conditions added until it stops refusing what the game allows, still refusing what
        it rightly refused — or None where no reading tells the two apart.

        A constraint that turns away a legal move is wrong as it stands, which is not the same as being worth
        nothing: everything it rightly refused would come back if it were simply dropped, and the next position
        would learn it all again. So it is given more to say, since a constraint that says more covers less, and
        the cases it must go on covering are what it may say it from.

        **It used to ask for a condition every kept case shares, and that is why it never worked.** Sharing is
        what makes adding a condition safe — lose nothing — and across boards the kept cases share almost
        nothing, so there was nothing to offer. Measured over ninety-five repairs in a day's runs, it mended one
        and refused to mend ninety-four; every one of those was then regrown from nothing, so a majority of the
        constraint set was being rebuilt each position while everything claimed it was being carried forward.

        **What it asks for now is what the job actually is: a condition that tells the kept cases from the move
        that must be released.** Of the conditions that release something, the one keeping most is taken. That
        can lose some of what the constraint refused — and losing some is the point, because the alternative was
        losing all of it. What is given up goes back to being unaccounted for, which is where regrowing left it
        too, and the next position learns it properly.

        None where nothing separates them at all, or where what survives refuses nothing: the constraint was
        wrong rather than merely too broad, and regrowing is the honest answer to that."""
        if not keeping:
            return None
        offered = [
            one
            for one in {literal for case in keeping for literal in case.held}
            if one not in clause.body
        ]
        found = clause
        kept = list(keeping)
        left = [one for one in without if self.covers(clause, one)]
        while left and offered:
            # Releasing at all comes first, since that is what makes it right; keeping most comes second, since
            # that is what makes it worth having. Sorted the other way round, the condition that keeps
            # everything and releases nothing wins every time and the loop never ends.
            best = max(
                offered,
                key=lambda one: (
                    sum(1 for case in left if one not in case.held),
                    sum(1 for case in kept if one in case.held),
                ),
            )
            if not any(best not in case.held for case in left):
                return None
            offered = [one for one in offered if one != best]
            found = Clause((*found.literals, best.denied), found.probability, found.name)
            left = [one for one in left if self.covers(found, one)]
            kept = [one for one in kept if self.covers(found, one)]
        if left or not kept:
            return None
        logger.debug(
            "Mended a constraint with %d more conditions: it releases %d moves the game allows and goes on "
            "refusing %d of the %d it did",
            len(found.body) - len(clause.body), len(without), len(kept), len(keeping),
        )
        return found

    def generalised(self, clause: Clause, example: Example) -> Clause | None:
        """The least general constraint covering both what that one covers and that case.

        This is the whole method in one operation and there is no search in it. Pair each reading of the clause
        with a reading of the case saying the same sort of thing; where the two agree on a term keep it, where
        they differ put a variable, and go inside a term to do the same to its parts. What is left covers both,
        and nothing more general does.

        **The variables come out by construction.** One map from each differing pair of terms to a variable serves
        the whole clause, so a case whose mover is one side and a case whose mover is the other give the *same*
        variable wherever either side appeared. That is the constraint about a player's own things, arrived at by
        generalising two cases rather than by searching for it.

        **The readings that pin the rest down are paired first.** A position offers one reading per parameter and
        sixty-four of the board, so pairing a board reading first would be choosing between sixty-four pairings
        with nothing to choose by. Pairing the parameters first fixes what their coordinates stand for, and a
        board reading at those coordinates then reuses those same variables rather than inventing new ones — which
        is what turns "the square at row four" into "the square the origin names"."""
        if clause.head is None:
            return None
        by_predicate = example.by_predicate
        shared: dict[tuple[Term, Term], Variable] = {}
        taken = self._taken(clause)
        kept: list[Literal] = []
        for literal in sorted(clause.body, key=lambda one: len(by_predicate.get(one.predicate, ()))):
            # A reading the case has word for word pairs with itself and survives whole. Asked of the case's own
            # set of readings it is one lookup; found by scoring the candidates it is sixty-three wasted walks and
            # then the answer. Two cases from one position agree about the whole board, so this is nearly every
            # reading nearly every time.
            if literal in example.held:
                kept.append(literal)
                continue
            best = self._best_pairing(literal, by_predicate.get(literal.predicate, ()), shared)
            if best is not None:
                kept.append(self._paired(literal, best, shared, taken))
        speaking = self._speaking(kept)
        if not speaking:
            return None
        return Clause((clause.head, *(one.denied for one in speaking)), clause.probability, clause.name)

    def both(self, one: Clause, other: Clause) -> Clause | None:
        """The least general constraint saying what both of those say, or None where they share nothing.

        The same operation as widening a constraint with a case, with a constraint on the other side instead of a
        case. It is how two constraints that were one constraint all along — learned once per side, or once per
        position — are found to be one, and it lives here so that going inside a term and sharing a variable
        across a whole clause are written once rather than twice."""
        if one.head is None or other.head is None or one.head != other.head:
            return None
        by_predicate: dict[str, list[Literal]] = {}
        for literal in other.body:
            by_predicate.setdefault(literal.predicate, []).append(literal)
        shared: dict[tuple[Term, Term], Variable] = {}
        taken = self._taken(one, other)
        kept: list[Literal] = []
        theirs = frozenset(other.body)
        for literal in sorted(one.body, key=lambda held: len(by_predicate.get(held.predicate, ()))):
            if literal in theirs:
                kept.append(literal)
                continue
            best = self._best_pairing(literal, by_predicate.get(literal.predicate, ()), shared)
            if best is not None:
                kept.append(self._paired(literal, best, shared, taken))
        speaking = self._speaking(kept)
        if not speaking:
            return None
        return Clause((one.head, *(held.denied for held in speaking)), one.probability, one.name)

    def covers(self, clause: Clause, example: Example, among: Sequence[Clause] = ()) -> bool:
        """Whether that constraint's body holds of that case, under some reading of its variables.

        The readings with nothing left open are asked first, and each is one lookup. Nearly every reading of a
        constraint grown from cases is of that sort, and a case failing any of them fails outright — so almost
        every case is turned away by a handful of lookups and never searched at all.

        `among` is the constraints this one is one of, and it is needed only where the body asks whether some
        other action would be allowed. What answers that is the layer below — those of them that ask no such
        question — and it is worked out from the set rather than declared anywhere."""
        open_ended = []
        for literal in clause.body:
            if literal.predicate in (ALLOWED, ALLOWED_AFTER, TAKEN_AFTER):
                open_ended.append(literal)
            elif literal.ground:
                if self._evaluable.evaluable(literal.predicate):
                    if not self._evaluable.holds(literal):
                        return False
                elif literal not in example.held:
                    return False
            else:
                open_ended.append(literal)
        if not open_ended:
            return True
        return self._holds(self._in_order(open_ended, example), example, Substitution(), among) is not None

    def _in_order(self, open_ended: list[Literal], example: Example) -> list[Literal]:
        """Those conditions in the order worth asking them.

        **The readings with the fewest ways of being matched go first.** A position offers one reading of where a
        move starts and sixty-four of the board, and asking the board first means trying sixty-four ways round of
        something the parameter would have settled outright. The answer is the same either way; the searching is
        not.

        **What is computed rather than looked up keeps its place, after all of it.** A condition comparing two
        distances can only be answered once something has said what they are, and a condition working a distance
        out can only be answered once the places it is between are settled. Sorting those to the front — which is
        where counting their matches would put them, since they match no reading at all — asks them before
        anything has bound their variables, and a question that cannot be answered is not answered yes. That is
        how a set of perfectly good rules quietly stops holding.

        **A hypothetical goes last of all, for that reason twice over.** It is answered by asking rather than by
        looking up, so counting its matches finds none and sorts it to the very front, where its variables stand
        for nothing and it answers no — the same silent failure, in the condition that matters most. And it is
        the one condition here that costs a board: drawing what the move does, reading the position it leads to,
        and putting every reply to the rules below. Asked after everything else, it is asked only of the cases
        that got that far."""
        if len(open_ended) < 2:
            return open_ended
        read = [one for one in open_ended if not self.computes(one.predicate)]
        read.sort(key=lambda one: len(example.by_predicate.get(one.predicate, ())))
        worked = [one for one in open_ended if self._evaluable.evaluable(one.predicate)]
        asked = [one for one in open_ended if one.predicate in (ALLOWED, ALLOWED_AFTER, TAKEN_AFTER)]
        return [*read, *worked, *asked]

    def _widened(
        self,
        left: Sequence[Example],
        index: CaseIndex,
        deadline: float,
        held: Sequence[Clause],
        positions: int,
        table: HypothesisTable | None = None,
        pool: Sequence[Example] = (),
    ) -> tuple[Clause | None, list[Example], Clause | None]:
        """One constraint, generalised from a case as far as what the game allows will let it go.

        It starts from a single case said outright, which covers that case and nothing else, and is generalised
        against further cases one at a time. Each generalisation covers strictly more, so the moment one would
        cover a candidate the game allows, that generalisation is refused and another case is tried instead.

        This is the opposite of growing a constraint downward from nothing, and the difference is where the work
        goes. Growing downward, every candidate condition is scored against every case, and there are thousands of
        candidates. Generalising upward, the cases say what the constraint is, and what the game allows is
        consulted once per generalisation rather than once per candidate condition.

        **Generalising never shortens, and that is measured.** It pairs each condition with a reading of the same
        sort and drops one only where no partner exists; between two whole boards a partner always exists. Of
        four thousand widenings offered to a hundred-and-sixty-one condition constraint, none failed for want of
        a widening, about half were refused for reaching a legal move, and the accepted ones came out at a
        hundred and fifty-seven. Length comes off in distilling or not at all."""
        wider = self._widest(left[0], held, index)
        if wider is not None:
            return wider
        start, clause = None, None
        for number, one in enumerate(left):
            if self._clock() >= deadline:
                break
            briefest = self._recalled(one, left, pool, index, deadline, table)
            if briefest is not None:
                start, clause = number, briefest
                break
        if start is None or clause is None:
            logger.debug("No case can be said briefly enough to refuse nothing the game allows")
            return None, [], None
        used = [left[start]]
        for example in (*left[:start], *left[start + 1 :]):
            if self._clock() >= deadline:
                break
            # A case the constraint already covers has nothing to widen it with: the least general thing covering
            # both is the constraint itself. Asking first is a handful of lookups where generalising is a walk
            # over every reading, and the more a constraint has been widened the more cases this skips — so the
            # work per case falls as the constraint gets better, rather than staying flat.
            if self.covers(clause, example):
                used.append(example)
                continue
            candidate = self.generalised(clause, example)
            if candidate is None or not candidate.body or self.slips(candidate, index):
                continue
            clause = candidate
            used.append(example)
        if not clause.body or not self._supported(used, positions):
            return None, used, None
        return clause, used, None

    def _widest(
        self, wanted: Example, held: Sequence[Clause], index: CaseIndex
    ) -> tuple[Clause, list[Example], Clause] | None:
        """A constraint already believed, widened to cover that case too, where one can be.

        Tried before anything new is begun. A game met a position at a time offers the same rule over and over in
        slightly different circumstances, and a learner starting fresh each time would answer with a new
        constraint for each, ending with as many rules as it had seen moves."""
        for clause in held:
            wider = self.generalised(clause, wanted)
            if wider is None or not wider.body or wider == clause:
                continue
            if self.slips(wider, index):
                continue
            logger.debug("Widened %s to refuse another case", clause.readable)
            return wider, [wanted], clause
        return None

    def slips(self, clause: Clause, index: CaseIndex) -> bool:
        """Whether that constraint would refuse anything the game allows.

        Said aloud because it is the one test a searcher elsewhere has to be able to make. Whatever proposes
        candidate bodies — here, or in another process — decides nothing else about them: a body that turns
        away a legal move is wrong wherever it was thought of.

        None is tolerated. Letting a refused candidate through is a move OMF proposes and the game rejects, which
        it finds out about at once; refusing an allowed one is a move OMF will never find, and nothing will ever
        tell it what it missed. Those are not the same mistake and are not priced the same.

        The cases are narrowed to those sharing a settled reading with the constraint before any is asked. No case
        the constraint covers is left out of that, since covering means having every settled reading of it."""
        # Computed rather than read, which now includes what is *asked* about a board that does not exist
        # yet. No case carries one, so narrowing by it finds no case at all and the constraint is declared
        # to turn nothing away — every body containing a question would pass the guard unexamined.
        settled = [one for one in clause.body if one.ground and not self.computes(one.predicate)]
        return any(self.covers(clause, one) for one in index.narrowed(settled))

    def _briefest(
        self,
        example: Example,
        left: Sequence[Example],
        index: CaseIndex,
        deadline: float,
        longest: int = 4,
    ) -> Clause | None:
        """The shortest thing sayable from that case which refuses it and refuses nothing the game allows.

        **Searched by size, because neither direction that hill-climbs finds it.** Generalising upward never
        shortens — it pairs a condition with a reading of the same sort and drops one only where no partner
        exists, so a clause born at a hundred and sixty conditions dies at a hundred and fifty-seven. Growing
        downward by taking the condition that releases most takes the most specific condition there is, every
        time, because within one position a coordinate separates maximally and a relation separates weakly: it
        gives `refused :- destination(1, 1)`, and when that is taken away, `refused :- self(1, 1)`. Asked by
        size, the first body that works is the shortest one, which is the property neither has.

        **What makes it safe is the guard and not the size.** A short clause is not a good clause — `self(1, 1)`
        is one condition — and this would prefer exactly those, were it judging against one position's twenty
        legal moves. It is judging against every legal move seen anywhere, so a rule that names a square dies on
        the first position where that square has a legal move out of it.

        **It does not stop at the first size that works, and that is measured.** Among bodies built from the
        same readings a longer one always refuses less, so stopping there looks safe — but a body whose every
        shorter part turns away a legal move is reached by no shorter search, and it may refuse a great deal
        more than anything short enough to be found. Over eight cases a run's constraints still left standing,
        judged against eighteen hundred legal moves, half lost between a third and a half of the coverage
        available one size further on. So the search keeps going `further` sizes past the first that works.

        Among everything standing in that window, the one refusing most of what is still unaccounted for *for
        what it costs to say* is taken. Coverage alone would take the longest body every time, since a longer
        body is reached only where its shorter parts slip, and those are the ones that refuse most.

        None where nothing of `longest` conditions or fewer will do. The caller then tries another case, and a
        case no brief rule fits is a case something longer has to account for. Where the clock runs out with
        something already standing, that is returned rather than nothing: it passed the same guard, and a
        constraint found is worth more to the caller than the search it did not finish."""
        offered = self.offered(example)
        standing: list[Clause] = []
        slipped: dict[int, set[frozenset]] = {}
        worked: int | None = None
        for size in range(1, longest + 1):
            if worked is not None and size > worked + self._further:
                break
            slipped[size] = set()
            for chosen in combinations(offered, size):
                if self._clock() >= deadline:
                    return self._worth(standing, left)
                body = frozenset(chosen)
                if not self.worth_trying(body, slipped.get(size - 1)):
                    continue
                clause = Clause((Literal(REFUSED, ()), *(one.denied for one in chosen)))
                if self.slips(clause, index):
                    slipped[size].add(body)
                else:
                    standing.append(clause)
            if standing and worked is None:
                worked = size
        return self._worth(standing, left)

    def offered(self, example: Example) -> tuple[Literal, ...]:
        """What a body may be built out of: the readings the candidate is in, and the questions it can be put.

        **Two kinds of condition, and until now only one of them was ever proposed.** A reading is produced
        when the position is read and lives in the case; a question is answered by computing when a clause is
        checked. Everything that checks a clause has handled both since it was written — `covers` dispatches
        them, `_in_order` puts them last for being dear, `computes` names them as a class. Nothing built one,
        because a body is assembled out of what a case carries and a question is not carried.

        The consequence was one class of rule outside the space and nothing saying so: the rules about what
        could happen next. A king may not be left where it can be taken; a king may not castle across a square
        an enemy could reach. Both were writable by hand and unfindable by search.

        **A reading said of a later moment is always about this candidate, so it is always offered.** What a
        case carries about now is narrowed by linkedness — a position says sixty-four things about a board and
        every candidate in it carries all of them identically, so they separate nothing. A reading of the board
        a candidate *leads to* is not like that: there is one such board per candidate, and it exists because
        of the candidate. Nothing has to be worked out about whether it is linked."""
        asking = self._hypothetical.askable(example) if self._hypothetical is not None else ()
        later = tuple(one for one in example.literals if one.when is not None)
        return (*self._readings.tied(example.literals), *later, *asking)

    def worth_trying(self, body: frozenset, slipped: set[frozenset] | None) -> bool:
        """Whether that body is worth putting to the guard at all, given what slipped one condition shorter.

        **A body containing a part that already passed is dominated and need never be asked about.** Conditions
        only narrow, so it refuses no more than that part does, and it costs more to say — whatever the price,
        the part wins. Only a body *every* shorter part of which turns away a legal move can beat what is
        already standing, and that is exactly the body no shorter search reaches.

        Said aloud beside `slips` because it is the other half of the same question and a searcher elsewhere
        has to be able to make it too. Where nothing shorter was recorded, everything is worth trying.

        This is what makes looking past the first working size affordable. Measured over twelve positions, the
        wider search cost twice the time a position and the arm never accumulated a rule set; asking the guard
        only about bodies whose every part slipped is the same search without the part of it that was never
        going to win. It is the mirror of what the hypothesis table already keeps slipping bodies *for*."""
        if slipped is None:
            return True
        return all(frozenset(part) in slipped for part in combinations(body, len(body) - 1))

    def _worth(self, standing: Sequence[Clause], left: Sequence[Example]) -> Clause | None:
        """Whichever of those refuses most of what is unaccounted for, for what it costs to say.

        None where nothing stands, which is the caller's signal to try another case."""
        if not standing:
            return None
        return max(standing, key=lambda one: sum(1 for case in left if self.covers(one, case)) / self.cost(one))

    def _recalled(
        self,
        example: Example,
        left: Sequence[Example],
        pool: Sequence[Example],
        index: CaseIndex,
        deadline: float,
        table: HypothesisTable | None,
        longest: int = 4,
    ) -> Clause | None:
        """The briefest thing that refuses that case, taken from what has already been tried where it can be.

        **Searching from nothing for every case is most of what the search does, and most of it has been done.**
        A body is built from a case's own readings, and within a position those readings are about the candidate
        rather than about the board — what stands where the mover stands, how far it goes — so the same body
        turns up again and again from different cases. Measured over thirteen cases of one position: three
        offers in five were bodies somebody had already tried, and by the thirteenth, three hundred and
        seventy-four things already held refused it before anything was searched at all.

        So the table is asked first. What comes back may be briefer than anything this case could have built,
        since it may have been found from a case with readings this one has not got — which is a better answer
        by the measure being applied and not merely a cheaper one.

        **It searches the same window the unshared search does**, `further` sizes past the first that works,
        and chooses among everything reaching this case by coverage over price rather than by taking the
        briefest. Two routes to the same rule that disagree about which rule is best are two learners, and
        which one an agent got would depend on whether a table happened to be passed.

        Without a table it searches, exactly as before. That is not a fallback kept for tidiness: it is the
        thing the table has to be shown to agree with."""
        if table is None:
            return self._briefest(example, left, index, deadline)
        held = [one for one in table.useful() if self.covers(one.clause, example)]
        if not held:
            # Nothing held reaches this case, so it is searched — and everything tried on the way is written
            # down, not only what won. What loses here is what another case will find already answered.
            offered = self.offered(example)
            worked: int | None = None
            for size in range(1, longest + 1):
                if worked is not None and size > worked + self._further:
                    break
                beyond = table.slipping(size - 1) if worked is not None else None
                table.tell(self._tester.tried(offered, size, pool, index, deadline=deadline, beyond=beyond))
                held = [one for one in table.useful() if self.covers(one.clause, example)]
                if held and worked is None:
                    worked = size
                if self._clock() >= deadline:
                    break
        if not held:
            return None
        return max(
            held,
            key=lambda one: sum(1 for case in left if self.covers(one.clause, case)) / self.cost(one.clause),
        ).clause

    def _outright(self, example: Example) -> Clause:
        """That case said as a constraint: everything read of it, and nothing else.

        The most specific constraint there is about it. It covers that case and, unless another case reads exactly
        the same, nothing else — which is where generalising starts from.

        **Nothing is left out for being true of every case.** A reading that holds of every candidate in a
        position excludes nothing *on its own*, and the temptation is to drop it — but what a position says of
        itself is exactly what becomes discriminating once a variable ties it to where the move is going. Drop the
        board and the only readings left are the parameters, and no constraint about what stands anywhere can ever
        be built. What is genuinely surplus comes out later, when the constraints are distilled.

        **Growing the constraint from the case instead was tried and made things worse.** Adding conditions one at
        a time, each releasing most of the moves the game allows, gives one-condition constraints that refuse
        nothing legal — and every one of them is a coordinate, because within a single position a coordinate
        separates maximally and what stands on a square separates weakly. `refused :- destination(1, 1)` is short
        and generalises to nothing. The length was never the disease."""
        return Clause((Literal(REFUSED, ()), *(one.denied for one in example.literals if not one.negated)))

    def _best_pairing(
        self, literal: Literal, candidates: Sequence[Literal], shared: dict[tuple[Term, Term], Variable]
    ) -> Literal | None:
        """Which of the case's readings to pair that one with: the pairing that keeps the most said outright, and
        of those the one inventing the fewest variables.

        **A reading the case has word for word is taken at once.** Nothing can beat it — every term survives and
        no variable is invented — so the remaining candidates are not looked at. A position read in full offers
        sixty-four board readings per case, most of them identical between two cases, and this is the difference
        between asking sixty-four questions and asking one.

        **Nothing is built to be scored.** A pairing is worth what its terms are worth, which is counted by
        walking them; building the reading it would give, only to throw it away for all but one candidate, was
        where nearly all of the time went."""
        best, score = None, None
        for candidate in candidates:
            if candidate.arity != literal.arity or not self._about_the_same(literal, candidate):
                continue
            if candidate == literal:
                return candidate
            added: set[tuple[Term, Term]] = set()
            held = (self._scored(literal.arguments, candidate.arguments, shared, added), -len(added))
            if score is None or held > score:
                best, score = candidate, held
        return best

    def _about_the_same(self, one: Literal, other: Literal) -> bool:
        """Whether two readings are about the same thing, so that generalising them means anything.

        A reading names what it is about and then says what was read of it. Where those naming places hold a
        *name* — a thing the game called something — two readings that name different things are not two accounts
        of one thing, and pairing them gives a variable standing for "either of these two unrelated places",
        which is true of everything and says nothing. The distance between the origin's row and the destination's
        row generalises with the same distance in another case, and never with the distance between a row and a
        column.

        Only names are held to this. Where a naming place holds a number — a cell's row, a list's index — it is
        a place that very much should generalise, since tying one case's row to another's is how every rule about
        where a thing may go is arrived at.

        **And a place holding a value is not a naming place, wherever it sits.** Usually the value is last and
        this takes care of itself; in two readings it is not, and which way one number lies from another was
        being frozen as though it named something. A thing that goes one way and a thing that goes the other were
        then two rules nothing could join — which is why every pawn rule is written twice and a diagonal takes
        two rules where it should take one. `CandidateReadings.VALUING` says where those places are."""
        valuing = VALUING.get(one.predicate, ())
        return all(
            number in valuing
            or not isinstance(mine, Constant)
            or not isinstance(theirs, Constant)
            or mine == theirs
            for number, (mine, theirs) in enumerate(zip(one.arguments[:-1], other.arguments[:-1]))
        )

    def _scored(
        self,
        mine: Sequence[Term],
        theirs: Sequence[Term],
        shared: dict[tuple[Term, Term], Variable],
        added: set[tuple[Term, Term]],
    ) -> int:
        """What pairing those terms would keep said outright, gathering the pairs it would have to invent a
        variable for. The same walk `_paired_term` does, counting instead of building."""
        total = 0
        for one, other in zip(mine, theirs):
            if one == other:
                total += self._settled_term(one)
            elif (
                isinstance(one, Functor)
                and isinstance(other, Functor)
                and one.name == other.name
                and len(one.arguments) == len(other.arguments)
            ):
                total += 1 + self._scored(one.arguments, other.arguments, shared, added)
            elif (one, other) not in shared:
                added.add((one, other))
        return total

    def _taken(self, *held: Clause | Example) -> set[str]:
        """The names already standing for something in those clauses.

        A variable introduced by generalising must not be called what something else is already called. A clause
        that has been folded once carries X1; fold it again and the new X1 is the old one, the agreement that
        should map the fold back to what it generalised contradicts itself, and the fold covers neither. Each
        reading of it holds on its own and no reading of them all holds together, which is what makes it look
        like anything but a name."""
        found: set[str] = set()
        for one in held:
            for literal in (one.body if isinstance(one, Clause) else one.literals):
                found.update(variable.name for variable in literal.variables)
        return found

    def _paired(
        self,
        one: Literal,
        other: Literal,
        shared: dict[tuple[Term, Term], Variable],
        taken: set[str] = frozenset(),  # type: ignore[assignment]
    ) -> Literal:
        """The two readings as one: their terms where they agree, a variable where they do not.

        The same pair of differing terms always gives the same variable, across the whole constraint. That is what
        makes one variable appear in two readings rather than two variables appearing in one each, and it is the
        difference between a constraint that says something and one that says nothing."""
        return Literal(
            one.predicate,
            tuple(
                self._paired_term(mine, theirs, shared, taken)
                for mine, theirs in zip(one.arguments, other.arguments)
            ),
            one.negated,
        )

    def _paired_term(
        self,
        mine: Term,
        theirs: Term,
        shared: dict[tuple[Term, Term], Variable],
        taken: set[str] = frozenset(),  # type: ignore[assignment]
    ) -> Term:
        """Two terms as one, going inside a term to do the same to its parts.

        A square holding a white rook and a square holding a white king agree on being a square and on the colour,
        and differ on the kind — so what they have in common is a square whose piece is white and of some kind,
        not a variable standing for the whole thing. Stopping at the top would throw away everything the two had
        in common the moment they were not identical."""
        if mine == theirs:
            return mine
        if (
            isinstance(mine, Functor)
            and isinstance(theirs, Functor)
            and mine.name == theirs.name
            and len(mine.arguments) == len(theirs.arguments)
        ):
            return Functor(
                mine.name,
                tuple(
                    self._paired_term(one, other, shared, taken)
                    for one, other in zip(mine.arguments, theirs.arguments)
                ),
            )
        held = shared.get((mine, theirs))
        if held is None:
            number = len(shared) + 1
            while FRESH.format(number=number) in taken:
                number += 1
            held = Variable(FRESH.format(number=number))
            shared[(mine, theirs)] = held
            taken.add(held.name)  # type: ignore[attr-defined]
        return held

    def _settled(self, literal: Literal) -> int:
        """How much of that reading is still said outright, counting inside its terms.

        What a generalisation is worth is how much of the two readings survived it, so the pairing that keeps the
        most is the pairing taken."""
        return sum(self._settled_term(one) for one in literal.arguments)

    def _settled_term(self, term: Term) -> int:
        if isinstance(term, Variable):
            return 0
        if isinstance(term, Functor):
            return 1 + sum(self._settled_term(one) for one in term.arguments)
        return 1

    def _speaking(self, literals: Sequence[Literal]) -> list[Literal]:
        """Those readings that ask for anything at all.

        A reading saying nothing outright, whose every variable is used nowhere else, asks only that the case have
        such a reading — which every case does, since they all come from one vocabulary. It restricts nothing, and
        a constraint carrying a dozen of them is a dozen times slower to ask about and no more selective.

        This is not dropping a reading because the cases suggest it is surplus. It is dropped because it cannot
        say anything, whatever the cases turn out to be, so nothing is risked by its going.

        **A variable shared with another reading is the whole point and is never surplus.** `destination(R, C)`
        says nothing by itself, and says that the destination is on the origin's row the moment `R` is the origin's
        row too. So what is asked is whether the reading is entirely lone variables, not whether its last place is
        one."""
        counted: dict[Variable, int] = {}
        for literal in literals:
            for variable in literal.variables:
                counted[variable] = counted.get(variable, 0) + 1
        return [
            literal
            for literal in literals
            if self._settled(literal) or any(counted.get(one, 0) > 1 for one in literal.variables)
        ]

    def _supported(self, used: Sequence[Example], positions: int) -> bool:
        """Whether those cases come from enough different positions to be evidence rather than coincidence.

        Distinct positions, not cases. A constraint tying four readings to one variable will hold of plenty of
        cases in the position it was grown from — the readings happened to agree there — and of none anywhere
        else. Counting cases cannot tell that apart from a rule; counting the positions they came from can,
        because a coincidence has to recur somewhere else to survive."""
        if not used:
            return False
        places = {one.where for one in used if one.where is not None}
        return len(places) >= positions if places else len(used) >= max(1, positions)

    def _could_take(self, literal: Literal, example: Example, among: Sequence[Clause]) -> bool:
        """Whether the other side, once this candidate is done, could take away the thing those terms name.

        **Said as what could happen and not as a place reached.** Arriving somewhere is the danger in chess and
        in checkers and not in a game where a card is turned or a score passes a mark. What is general is that
        something could happen next that one would rather did not, and a change is how a happening is said here.

        The rules that answer are those asking no such question themselves, which is what keeps it to one ply:
        put to all of them, whether their reply is allowed would ask whether it leaves their own king safe,
        which asks about mine."""
        if self._hypothetical is None or len(literal.arguments) < 2:
            return False
        whose, what = (self._plain(one) for one in literal.arguments[:2])
        if whose is None or what is None:
            return False
        below = self._hypothetical.below(among)
        return self._hypothetical.taken(
            example, whose, what, lambda case: any(self.covers(one, case, below) for one in below)
        )

    def _plain(self, term: Term) -> Value | None:
        """What that term stands for, or None where it stands for nothing settled."""
        if isinstance(term, Number):
            return term.value
        return term.name if isinstance(term, Constant) else None

    def _allowed(
        self, literal: Literal, example: Example, among: Sequence[Clause], afterwards: bool = False
    ) -> bool:
        """Whether the action those terms name would be allowed, in this same position, by the rules below.

        **Below, and never all of them.** Put to every rule, the question does not terminate: whether the enemy
        may take my king would depend on whether their king is safe, which depends on mine. Put to the rules that
        ask no such question themselves, it is answered in one step and cannot come back round.

        Where no hypothetical was built — a term standing for nothing settled, or the terms not making an action
        at all — the answer is that it does not hold. A question that cannot be put is not a question answered
        yes."""
        if self._hypothetical is None:
            return False
        where = example.where
        if afterwards:
            # The board this candidate leads to, which is the predictor's answer and not the constraints'. Where
            # it cannot say — nothing learned yet, or a consequence that will not draw here — there is no board
            # to ask about, and the question goes unanswered rather than answered yes.
            where = self._hypothetical.after(example)
            if where is None:
                return False
        case = self._hypothetical.case(where, literal.arguments)
        if case is None:
            return False
        below = self._hypothetical.below(among)
        return not any(self.covers(one, case, below) for one in below)

    def _computed(self, literal: Literal, agreed: Substitution) -> Substitution | None:
        """The agreement with that literal's last place standing for what it comes to, or None.

        Only the last place may be open, and only for a predicate that can work it out. Anything else left open is
        a question this cannot answer rather than one it answers wrongly, so it says so."""
        if not self._evaluable.computes(literal.predicate) or any(
            not one.ground for one in (Literal(literal.predicate, literal.arguments[:-1]),)
        ):
            return None
        found = self._evaluable.value(literal)
        if found is None:
            return None
        last = literal.arguments[-1]
        if isinstance(last, Variable):
            return agreed.bound(last, Number(int(found) if found == int(found) else found))
        return agreed if self._evaluable.holds(literal) else None

    def _holds(
        self,
        body: Sequence[Literal],
        example: Example,
        agreed: Substitution,
        among: Sequence[Clause] = (),
    ) -> Substitution | None:
        """A reading of the variables under which every one of those holds of the case, or None."""
        if not body:
            return agreed
        first, rest = agreed.applied(body[0]), body[1:]
        if first.predicate in (ALLOWED, ALLOWED_AFTER):
            held = self._allowed(first, example, among, first.predicate == ALLOWED_AFTER)
            return self._holds(rest, example, agreed, among) if held else None
        if first.predicate == TAKEN_AFTER:
            return self._holds(rest, example, agreed, among) if self._could_take(first, example, among) else None
        # `among` is carried through every branch, not only the ones that use it here. It is the set a
        # hypothetical's layer is worked out from, and a condition further along the body may be one — so
        # dropping it is not losing an unused argument, it is asking the layered question with nothing in the
        # layer. `below(())` is empty, nothing below refuses the reply, and "could they take the king?" comes
        # back yes for every move there is. It answered yes, which is the one way a question that cannot be put
        # must never be answered, and it did it silently: king safety refused all twenty legal moves of the
        # opening position and looked like a rule that was simply too strong.
        if self._evaluable.evaluable(first.predicate):
            if first.ground:
                return self._holds(rest, example, agreed, among) if self._evaluable.holds(first) else None
            # Its last place is what it comes to, where the others are settled — so a distance can be had and
            # then compared with another, which is what every rule about how a thing travels turns out to be.
            settled = self._computed(first, agreed)
            return None if settled is None else self._holds(rest, example, settled, among)
        if first.ground:
            return self._holds(rest, example, agreed, among) if first in example.held else None
        for literal in example.by_predicate.get(first.predicate, ()):
            found = self._unifier.unify(first, literal)
            if found is None:
                continue
            settled = self._holds(rest, example, agreed.then(found), among)
            if settled is not None:
                return settled
        return None
