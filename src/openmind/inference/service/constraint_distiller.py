import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence

from openmind.inference.model.case_index import CaseIndex
from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.refusal_learner import RefusalLearner
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal

logger = logging.getLogger(__name__)


class ConstraintDistiller:
    """The simplest set of constraints refusing exactly what these refuse.

    One thing is minimised — what the constraints cost — under what may not be given up, which is set out below.
    The one way of getting cheaper is to say several constraints as the one they are all instances of, which is
    what 1+1=2, 1+2=3 and 1+3=4 have to say once they are x+y=z.

    **A constraint's cost grows faster than its length**, and that is what asks for simplicity rather than merely
    permitting it. Priced by the reading, one constraint of ten readings and two of five cost the same and nothing
    prefers either; priced by the square they are a hundred against fifty, and the pair of simple ones wins. It
    does not therefore break everything into pieces: one of three beats three of three, nine against twenty-seven.
    Simple where it can be, few where it can.

    **This is where a lookup table loses.** A constraint and a table of the candidates it refuses both answer
    every case they were fit to, so nothing in the answers tells them apart. What tells them apart is that the
    constraint stays the same size as evidence accumulates and the table grows with it, and the only place that
    difference can be felt is a price on saying things.

    **What may not change is held from both sides.** A cheaper way of saying them must still refuse everything
    they refused, and must still refuse nothing the game allows. Neither alone is enough: asked only to keep
    refusing what it refused, the cheapest answer is the constraint with no conditions, which refuses everything;
    asked only to refuse nothing legal, the cheapest answer is no constraints at all.

    **Between those two, it is free to refuse more, and that freedom is the point.** Five constraints each about
    one pawn on one square fold into one about pawns, and the one refuses candidates that none of the five did —
    which is what a rule does and a lookup table cannot. Forbidding that, by asking for the same answers case for
    case, is forbidding every generalisation there is: it would keep the five and call the pile correct.

    **What it may wrongly refuse, it may wrongly refuse only in a position not yet seen.** The guard is every
    legal move the game has ever shown, so a fold is checked against all of them and not only against the
    position at hand. That still leaves a fold able to refuse something legal in a position nobody has met, which
    is the price of learning a rule from cases rather than being told one. It is recoverable: a constraint caught
    refusing a legal move has conditions added until it stops, or is dropped where no reading tells the cases
    apart.

    It keeps nothing: built once, it is given the constraints and the cases on every call."""

    def __init__(
        self,
        learner: RefusalLearner | None = None,
        clock: Callable[[], float] = time.monotonic,
        regrowing: float = 1.0,
    ) -> None:
        self._learner = RefusalLearner() if learner is None else learner
        self._clock = clock
        #: How long growing one constraint again may take. It is asked once per condition that will not come
        #: off, so it is a second rather than a budget, and a constraint it cannot regrow in that keeps what
        #: it had.
        self._regrowing = regrowing

    def distilled(
        self,
        clauses: Sequence[Clause],
        refused: Sequence[Example],
        allowed: Sequence[Example],
        priced: Callable[[Clause], float] | None = None,
        exponent: float = 2.0,
        per_clause: float = 1.0,
        budget: InferenceBudget | None = None,
        coverage: Mapping[Clause, frozenset[int]] | None = None,
    ) -> tuple[Clause, ...]:
        """Those constraints said as simply as they can be while answering every case exactly as before.

        `per_clause` is what having a constraint at all costs, before a word of it is said. Without it, a set is
        priced only by what its constraints say, and a hundred narrow constraints that each account for one
        position are no dearer than the one constraint they are all instances of — so nothing prefers the rule to
        the pile. It is the caller's number because how much a further rule costs is a fact about what the rules
        are for, not about the rules."""
        price = priced if priced is not None else (lambda one: per_clause + float(len(one.body)) ** exponent)
        seconds = budget.seconds if budget is not None else float("inf")
        # Folding and stripping get a share each, and folding may not spend the stripping's.
        #
        # Left to take what it liked, folding took everything: every constraint against every other grows with
        # the square of how many there are, so at a hundred and ninety it filled ten minutes and then some,
        # the deadline passed, and the strip pass — which asks the clock before it starts — did nothing. The
        # constraints kept their boards, every question asked of them stayed slow, and learning could no
        # longer grow even one new constraint inside its own budget. Three quarters of an hour a position to
        # stand exactly still, and all of it from a pass that never ran.
        folding = self._clock() + seconds / 2
        deadline = self._clock() + seconds
        guard = CaseIndex(tuple(allowed))
        # Folded first and stripped after, and it is worth saying why the other order was tried and dropped.
        # Stripping first is far faster — a position in thirteen seconds against forty — because short
        # constraints are cheap to fold. It also destroys the learning: stripped before anything is folded,
        # every constraint comes down to the one condition the guard happens not to object to, and a position
        # ends up accounted for by twenty-seven statements of the form "refused where the move starts at a1",
        # each true of that position and of nothing else. Measured against the rules there are to find, that
        # ordering scored none of seventy-two where this one scores thirty-one.
        found = list(clauses)
        # The guard is asked on every candidate way of saying the constraints, and it grows with every
        # position. Indexed once, a constraint is asked only of the legal moves that share a settled reading
        # with it, so the guard costs what it is worth rather than what it weighs.
        # What the learner already worked out, where it was handed over. Asking every constraint about every
        # candidate is the largest fixed cost here and it is the same answer either service computes.
        covered = self._covered(found, refused) if coverage is None else dict(coverage)
        cost = sum(price(one) for one in found)
        started, stopped = cost, "nothing further pays"
        # Folds first and in the order most likely to pay, then the conditions a constraint is no worse for
        # losing. Folding is what turns a pile of accounts of one rule into the rule; taking conditions off is
        # what makes each account cheap. Doing the cheap thing first leaves the constraints looking so unalike
        # that no fold can find them, which is the one order that arrives nowhere.
        for kind, offered in (("fold", self._folds(found, price, folding)), ("removal", None)):
            if offered is None:
                offered = iter(sorted(self._removals(found, price), key=lambda held: held[0]))
            consumed: set[Clause] = set()
            for delta, standing, dropped, how, build in offered:
                if self._clock() >= (folding if kind == "fold" else deadline):
                    stopped = "the time ran out"
                    break
                if delta >= 0:
                    continue
                if any(one in consumed for one in dropped):
                    continue
                if self._slips(standing, guard):
                    continue
                if not self._keeping(covered, standing, dropped, refused):
                    continue
                found = [one for one in found if one not in dropped] + list(standing)
                covered = {
                    **{held: where for held, where in covered.items() if held not in dropped},
                    **self._covered(standing, refused),
                }
                consumed.update(dropped)
                cost += delta
                logger.debug("%s, leaving %d constraints at a cost of %.0f", how, len(found), cost)
        found = self._stripped(
            found, guard, deadline, refused=refused, everywhere=self._everywhere(refused, allowed)
        )
        found, covered = self._without_the_surplus(found, self._covered(found, refused), price)
        cost = sum(price(one) for one in found)
        logger.info(
            "Distilled %d constraints costing %.0f into %d costing %.0f, refusing none of the %d moves the game "
            "has allowed: %s",
            len(clauses), started, len(found), cost, len(allowed), stopped,
        )
        return tuple(found)

    def _without_the_surplus(
        self,
        clauses: Sequence[Clause],
        covered: Mapping[Clause, frozenset[int]],
        price: Callable[[Clause], float],
    ) -> tuple[list[Clause], dict[Clause, frozenset[int]]]:
        """Those of them that are doing anything, which is not the same question as whether they are alike.

        **A constraint refusing only what the others already refuse is surplus, however it is written.** Folding
        asks whether two constraints are instances of one rule, and taking conditions off asks whether a
        constraint says more than it needs — neither notices two constraints that arrived by different routes and
        refuse exactly the same candidates. Nine of forty-six did, and every one was paid for in full on every
        question asked of the set.

        The dearest goes first, so what is kept is the cheapest way of refusing what was refused. Nothing is lost
        by construction: a constraint is dropped only where every candidate it refused is refused by one that
        stays — and a constraint refusing nothing among the cases in hand is left alone, since what it refuses is
        somewhere these cases are not."""
        found = sorted(clauses, key=price, reverse=True)
        kept = list(found)
        for clause in found:
            others = frozenset().union(
                *(covered[one] for one in kept if one is not clause), frozenset()
            )
            # A constraint refusing nothing here is not surplus, it is untested. Its cases are somewhere else —
            # another position, another game walked by another worker — and an empty set is contained in every
            # set, so treating the two alike throws away everything learned anywhere but here.
            if covered[clause] and covered[clause] <= others:
                kept = [one for one in kept if one is not clause]
        if len(kept) < len(clauses):
            logger.info(
                "Dropped %d constraints refusing nothing the others did not", len(clauses) - len(kept)
            )
        return kept, {one: covered[one] for one in kept}

    def _everywhere(
        self, refused: Sequence[Example], allowed: Sequence[Example]
    ) -> frozenset[Literal] | None:
        """The readings every case in hand has, which are what a constraint is carrying rather than saying.

        Within one position the whole board is like this: every candidate reads the same sixty-four squares, so
        a constraint grown there carries all of them and none of them is why anything was refused. They are what
        makes folding unaffordable — two constraints from two positions share nothing while each carries its own
        board — and taking them off first is what lets the rest of distilling happen at all.

        It is not a licence to take them off unchecked. A constraint without them covers more and may cover a
        legal move, which the guard still answers. This only says which removals are worth *trying* first."""
        if not refused:
            return None
        found = {one for one in refused[0].literals if one.ground}
        for example in (*refused[1:], *allowed):
            found &= example.held
            if not found:
                break
        return frozenset(found)

    def _stripped(
        self,
        clauses: Sequence[Clause],
        guard: CaseIndex,
        deadline: float,
        only: frozenset[Literal] | None = None,
        refused: Sequence[Example] = (),
        everywhere: frozenset[Literal] | None = None,
    ) -> list[Clause]:
        """Each constraint with every condition taken out that it refuses the same moves without.

        **Taking a condition out is the one change that cannot lose anything.** A shorter body is satisfied by
        everything the longer one was satisfied by, so what the constraint refuses can only grow — which is why
        this needs the guard and nothing else, while a fold, which is not monotone in that way, needs its
        coverage proved. Separating them is what makes this affordable: one pass per constraint and one question
        per condition, against a handful of legal moves, with nothing global to re-work out.

        **Done first, and it is a trade rather than a free choice.** Stripped first, a constraint may lose the
        conditions tying it to where the move starts, and two constraints so stripped no longer look alike in
        the way that would show they were one constraint — so some folds are lost. Folded first, the folds
        take every second there is and nothing is ever stripped, which leaves every constraint carrying a
        whole position. The first is the order that arrives somewhere, and what it gets wrong is the kind of
        wrong the next position catches: a constraint stripped too far refuses a move the game allows, and is
        then repaired or dropped.

        A condition kept is kept because taking it out would refuse a move the game allows. That is the whole of
        why anything stays, and it is why the guard has to be every legal move ever seen rather than this
        position's alone."""
        found: list[Clause] = []
        taken = 0
        for number, clause in enumerate(clauses):
            if self._clock() >= deadline:
                found.extend(clauses[number:])
                break
            kept = clause
            for literal in clause.body:
                if only is not None and literal not in only:
                    continue
                without = Clause(
                    tuple(one for one in kept.literals if one != literal.denied), kept.probability, kept.name
                )
                if without.body and not self._slips((without,), guard):
                    kept = without
            taken += len(clause.body) - len(kept.body)
            found.append(kept)
        if taken:
            logger.info(
                "Took %d conditions off %d constraints, refusing the same moves without them", taken, len(clauses)
            )
        return found

    def _slipping(self, clause: Clause, guard: CaseIndex) -> list[Example]:
        """The moves the game allows that this constraint would refuse, if any."""
        settled = [one for one in clause.body if one.ground and not self._evaluable(one)]
        return [one for one in guard.narrowed(settled) if self._learner.covers(clause, one)]

    def _postcodes(self, clause: Clause, everywhere: frozenset[Literal]) -> int:
        """How much of that constraint says which position it was fitted to rather than why a move is refused.

        **Every one of these is kept for the wrong reason, and nothing here can fix it.** Take a castling right
        out of a constraint and it reaches into an earlier position and refuses something legal there, so the
        guard keeps it — and measured, twenty-seven of twenty-seven were kept for that reason and none because
        the reading told two candidates apart where the constraint was grown. They do not say why a move is
        refused; they say which position it was fitted to.

        Three ways of getting rid of them were tried and all three failed, which is worth writing down so they
        are not tried again:

        - *Adding a condition instead* (`RefusalLearner.repaired`) draws from readings the kept cases all share,
          and within a position diverse candidates share only the position — so it can only trade one postcode
          for another.
        - *Growing the constraint again* from the cases that made it need the postcode starts from a case said
          outright, which is the whole board, and a short budget cannot widen it back down: forty-three
          constraints of seventeen hundred readings against forty-two of two hundred and ninety.
        - *Stripping harder* is what keeps them: by the guard's measure they are load-bearing, and the guard is
          right.

        What they mean is that the constraint is over-general and the reading that would make it right is not
        available — either because the evidence that would find it has not been seen, or because the concept
        is not in the vocabulary. This counts them so that is visible rather than invisible."""
        return sum(1 for one in clause.body if one in everywhere)

    def _covered(
        self, clauses: Sequence[Clause], refused: Sequence[Example]
    ) -> dict[Clause, frozenset[int]]:
        """Which of the refused candidates each constraint refuses, by their place in the list.

        Worked out once and kept up to date as constraints change, because the question asked of every candidate
        way of saying them — is anything that was refused let through — is answered from these in set lookups
        rather than by asking thousands of candidates again."""
        return {
            clause: frozenset(
                number for number, one in enumerate(refused) if self._learner.covers(clause, one, clauses)
            )
            for clause in clauses
        }

    def _keeping(
        self,
        covered: Mapping[Clause, frozenset[int]],
        standing: Sequence[Clause],
        dropped: Sequence[Clause],
        refused: Sequence[Example],
    ) -> bool:
        """Whether everything refused before is still refused after.

        **Only the candidates at risk are asked about.** Taking constraints away can only let through what those
        constraints were refusing, and what everything else refuses is unchanged — so the question is whether the
        few the dropped ones were alone in refusing are picked up by what replaces them. Asked of all four
        thousand, this is the whole cost of distilling; asked of the handful at risk, it is nothing."""
        at_risk = frozenset().union(*(covered.get(one, frozenset()) for one in dropped))
        held = frozenset().union(
            *(where for clause, where in covered.items() if clause not in dropped), frozenset()
        )
        for number in at_risk - held:
            if not any(self._learner.covers(one, refused[number], standing) for one in standing):
                return False
        return True

    def _slips(self, standing: Sequence[Clause], guard: CaseIndex) -> bool:
        """Whether any of those newly said constraints refuses a move the game allows.

        Only the constraints that changed are asked. The others said the same thing before the change as after,
        and they were standing then.

        **This is half of the check and not the whole of it.** It was written as the whole, on the reasoning that
        a shorter body and a fold both refuse a superset of what they refused before, so nothing refused could
        stop being refused. That reasoning is wrong about the fold: two constraints said as one does *not*
        reliably cover what each covered, and a set distilled on that assumption collapsed into a single
        constraint refusing nothing at all. So what must go on being refused is asked about too, by the caller,
        and the day that fold is a true generalisation this can go back to being the whole of it."""
        return any(
            self._learner.covers(clause, one)
            for clause in standing
            for one in guard.narrowed(
                [held for held in clause.body if held.ground and not self._evaluable(held)]
            )
        )

    def _evaluable(self, literal: Literal) -> bool:
        """Whether that reading is answered by computing rather than by a case having it, in which case no
        case can be narrowed to by it."""
        return self._learner.computes(literal.predicate)

    def _folds(
        self, clauses: Sequence[Clause], price: Callable[[Clause], float], deadline: float
    ) -> Iterator[tuple[float, tuple[Clause, ...], tuple[Clause, ...], str, Callable[[], list[Clause]]]]:
        """Ways of saying several constraints as the one they are instances of, the most promising first.

        **Which pairs are worth trying is guessed at before any of them is tried.** Folding two constraints means
        walking every reading of one against every reading of the other, and on constraints carrying a whole
        position that is the entire cost of distilling — so doing it for all of them, to find out which were worth
        doing, spends the budget on the answer rather than on the work.

        What is guessed by is how much the two already say alike: the readings they hold in common, counted by set
        intersection, which costs nothing. Two constraints sharing sixty readings are two accounts of one rule and
        fold into something that keeps them; two sharing none will fold into something that says almost nothing
        and be refused. Ordering by it means the folds that pay are reached in the first seconds rather than the
        last.

        **It orders and never forbids.** Every pair is still offered, and what is reached is decided by the
        budget, as everything else here is. A guess that puts a good fold late costs time; it cannot make a wrong
        set of constraints."""
        bodies = [frozenset(one.body) for one in clauses]
        alike = sorted(
            (
                (len(bodies[first] & bodies[second]), first, second)
                for first in range(len(clauses))
                for second in range(first + 1, len(clauses))
            ),
            key=lambda held: -held[0],
        )
        for _, first, second in alike:
            if self._clock() >= deadline:
                return
            running, taken = clauses[first], {first}
            best: tuple[float, Clause, frozenset[int]] | None = None
            for other in (second, *self._nearest(bodies, {first, second})):
                if self._clock() >= deadline:
                    return
                both = self._learner.both(running, clauses[other])
                if both is None or not both.body:
                    break
                running, taken = both, taken | {other}
                delta = price(running) - sum(price(clauses[one]) for one in taken)
                if best is None or delta < best[0]:
                    best = (delta, running, frozenset(taken))
            # One offer per chain, and it is the whole of the fold rather than its first step. Offered step by
            # step, the pair is taken and the three are then refused for being made of constraints already
            # spoken for — so four accounts of one rule become two accounts of it instead of one, which is
            # exactly the saving worth having thrown away for the sake of the smaller one.
            if best is not None:
                delta, folded, held = best
                yield (
                    delta,
                    (folded,),
                    tuple(clauses[one] for one in held),
                    f"said {len(held)} constraints as one",
                    lambda where=held, one=folded: [
                        clause for place, clause in enumerate(clauses) if place not in where
                    ] + [one],
                )

    def _nearest(self, bodies: Sequence[frozenset[Literal]], taken: set[int]) -> Iterator[int]:
        """The constraints most like the ones already folded, nearest first, so a fold of many is built out of the
        ones with most to give rather than out of whatever came next in the list."""
        left = sorted(
            (place for place in range(len(bodies)) if place not in taken),
            key=lambda place: -len(bodies[place] & bodies[min(taken)]),
        )
        yield from left

    def _removals(
        self, clauses: Sequence[Clause], price: Callable[[Clause], float]
    ) -> Iterator[tuple[float, tuple[Clause, ...], tuple[Clause, ...], str, Callable[[], list[Clause]]]]:
        """Ways of saying one constraint with a condition fewer.

        Cheap to offer and cheap to price, so all of them are, and the cheapest is taken first."""
        for number, clause in enumerate(clauses):
            was = price(clause)
            for literal in clause.body:
                without = Clause(
                    tuple(one for one in clause.literals if one != literal.denied), clause.probability, clause.name
                )
                if not without.body:
                    continue
                yield (
                    price(without) - was,
                    (without,),
                    (clause,),
                    f"said a constraint without {literal.predicate}",
                    lambda place=number, one=without: [*clauses[:place], one, *clauses[place + 1 :]],
                )
