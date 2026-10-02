import logging
from collections.abc import Sequence

import numpy as np

from openmind.inference.model.expression import Expression
from openmind.inference.model.pattern import Pattern
from openmind.inference.model.pattern_condition import PatternCondition
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.constant.inference_constant import ME, OTHER
from openmind.rbs.model.position_row import PositionRow
from openmind.structure.model.grid import Grid

logger = logging.getLogger(__name__)


class TermAssembler:
    """Terms built from the positions a heuristic is most wrong about, rather than searched for.

    **Growing a term one clause at a time cannot reach the terms a player would name.** A passed pawn is a
    pawn of mine and the absence of an enemy pawn over the squares ahead of it: two conditions and twenty-one
    denials, forty-four clauses in all. A search that grows one clause at a time multiplies its candidates
    every generation and is paid only for what predicts, and no three-clause fragment of a passed pawn
    predicts anything — so the term is sayable and unreachable, which is the worst a term can be.

    **So it is assembled against what is not yet explained.** `ClauseLearner` does this for the rules of a
    game: it takes one case, says what was read of it, drops what was read the same way of every case, and
    widens. The same move works here with the residual in place of legality — a condition earns its way into
    the term by making what the heuristic misses easier to predict, and a condition that does nothing for that
    is never added. Each clause is chosen because it helps, which is the gradient a blind search does not have.

    **What it assembles is a candidate and never a rule.** It gives back expressions, so an assembled term
    goes into the same search, the same economy and the same fit as anything the generator offered: a signal
    still has to stake a budget on it and the fit still has to keep it. Nothing here writes into a ruleset.

    **The anchor stays the residual, whoever supplies it.** Given what the games paid less what the fit reads,
    it assembles terms explaining what the payoff knows and the heuristic does not; given a teller's reading
    less the fit's, it assembles terms explaining where the heuristic and the teller part company. One
    argument, one code path, and what a teller is stays the caller's business."""

    def assembled(
        self,
        rows: Sequence[PositionRow],
        missed: Sequence[float],
        vocabulary: Vocabulary,
        how_many: int = 1,
        widest: int = 6,
        longest: int = 4,
        idioms: Sequence[tuple[str, str]] = (),
        says: object = None,
    ) -> tuple[Expression, ...]:
        """Terms that account for what `missed` holds, most telling first; none where nothing accounts for it.

        `missed` is one number per row: what the heuristic does not yet explain about it. Rows and misses are
        read together, so a caller handing a different number of each is asking something it has not said."""
        if len(rows) != len(missed):
            raise ValueError(f"{len(rows)} rows and {len(missed)} misses are not the same positions")
        wanted = np.asarray(missed, dtype=float)
        if len(rows) < 2 or not np.isfinite(wanted).any() or float(np.nanstd(wanted)) == 0.0:
            return ()
        found = [
            built for anchor in sorted(vocabulary.grids)
            if (built := self._over(anchor, rows, wanted, vocabulary, widest, longest, idioms, says)) is not None
        ]
        found.sort(key=lambda one: -one[1])
        return tuple(one for one, _ in found[:how_many])

    def _over(
        self,
        anchor: str,
        rows: Sequence[PositionRow],
        wanted: np.ndarray,
        vocabulary: Vocabulary,
        widest: int,
        longest: int,
        idioms: Sequence[tuple[str, str]],
        says: object,
    ) -> tuple[Expression, float] | None:
        """One term assembled over that anchor with what it accounts for, or None where nothing it says helps."""
        places = self._places(anchor, rows)
        if places is None:
            return None
        offered = self._offered(anchor, places, rows, vocabulary)
        offered.extend(self._idioms(idioms, places, rows, says))
        if not offered:
            return None
        masks = np.array([mask for _, mask in offered])
        conditions = [condition for condition, _ in offered]
        # Which conditions can be denied together: the same place read on two different bases. That is what
        # "no pawn of theirs here" is, and denying two values of one base says nothing — a thing is not two
        # values at once, so the pair never holds and its denial always does.
        pairs = [
            (one, other)
            for one in range(len(conditions))
            for other in range(one + 1, len(conditions))
            if conditions[one].steps == conditions[other].steps and conditions[one].base != conditions[other].base
        ]
        # **Rows held back, because a clause that helps here may help nowhere else.** Greedy assembly adds a
        # clause whenever it improves what it can see, and what it can see is the rows it is assembling from —
        # measured, that gave sixty-five clauses correlating at -0.08 on rows it had not seen. A clause is kept
        # only where it also improves on rows held out, which is what `HeuristicPonderer._split` does to choose
        # a price and for the same reason.
        every = np.arange(len(rows))
        assembling, back = every[every % 3 != 0], every[every % 3 == 0]
        if not len(back) or float(np.std(wanted[back])) == 0.0 or float(np.std(wanted[assembling])) == 0.0:
            assembling, back = every, every
        # **Several part-built terms carried at once, because the useful clause is often half of a pair.**
        # A knight guarded by a pawn is a knight here and a pawn at an offset, and neither half says anything
        # about how many such knights there are — so an assembly that commits to its best single clause picks
        # the wrong one and never recovers. Measured: that term came out at 0.000 against what it was meant to
        # find. Carrying the runners-up costs the width in work and lets the pair be found on the second step.
        beam = [(np.ones(masks.shape[1], dtype=bool), (), (), 0.0)]
        # **Every term the beam ever held, because the best one is usually the one that stopped growing.**
        # A beam built only from what could be extended drops a term the moment nothing improves it — which
        # is exactly when it is finished. Measured: a one-condition term reading its target perfectly was
        # thrown away for three-condition terms correlating at 0.169, and it read as the assembly ignoring
        # the vocabulary it had been given.
        settled = list(beam)
        while True:
            grown = []
            for held, kept, absences, best in beam:
                for one, told in self._grown(
                    masks, held, pairs if kept else (), wanted, len(rows), best, assembling, back, widest
                ):
                    if isinstance(one, tuple):
                        grown.append((self._held(masks, held, one), kept, (*absences, one), told))
                    else:
                        grown.append((self._held(masks, held, one), (*kept, one), absences, told))
            if not grown:
                break
            grown.sort(key=lambda one: -one[3])
            beam = grown[:widest]
            settled.extend(beam)
            # **Four conditions a round, not forty-four in one.** Every search in the literature is capped
            # between four and six, and nothing is reported to recover a target longer than that — so length
            # is not what this is for. A term that earns its keep is named and offered back as one condition,
            # and the next round assembles over the larger vocabulary. Depth comes from the layers.
            if max(len(one[1]) + len(one[2]) for one in beam) >= longest:
                break
        held, kept, absences, best = max(settled, key=lambda one: one[3])
        if not kept:
            return None
        pattern = Pattern(
            anchor,
            tuple(conditions[at] for at in kept),
            tuple((conditions[one], conditions[other]) for one, other in absences),
        )
        logger.info(
            "Assembled a term of %d conditions and %d denials over %s, which accounts for %.3f of what the "
            "heuristic was missing",
            len(kept), len(absences), anchor, best,
        )
        return Expression(self._template(pattern, vocabulary), len(kept) + 2 * len(absences), 0, pattern), best

    def _grown(
        self,
        masks: np.ndarray,
        held: np.ndarray,
        pairs: Sequence[tuple[int, int]],
        wanted: np.ndarray,
        rows: int,
        best: float,
        assembling: np.ndarray,
        back: np.ndarray,
        widest: int,
    ) -> list[tuple[object, float]]:
        """The additions that account for most of what is missed, best first; empty where none accounts for more.

        **A condition and the denial of a pair are weighed against each other**, because which of them a term
        needs is not something to settle in advance: a passed pawn is one condition and twenty-one denials, a
        doubled pawn two conditions and none.

        **Chosen on the rows it assembles from and kept only where the rows held back agree.** Greedy assembly
        adds whatever improves what it can see, and what it can see is its own rows — measured, that gave
        sixty-five clauses correlating at -0.08 on rows it had never read."""
        found: list[tuple[object, float]] = []
        for offers, how in ((masks & held, None), (self._denied(masks, held, pairs), pairs)):
            if offers is None or not len(offers):
                continue
            counted = self._counted(offers, rows)
            told = self._told_of(counted[:, assembling], wanted[assembling])
            proven = self._told_of(counted[:, back], wanted[back])
            for at in np.argsort(-told)[:widest]:
                if told[at] > 0.0 and proven[at] > best:
                    found.append(((how[int(at)] if how is not None else int(at)), float(proven[at])))
        found.sort(key=lambda one: -one[1])
        return found[:widest]

    def _denied(self, masks: np.ndarray, held: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray | None:
        if not pairs:
            return None
        return np.array([held & ~(masks[one] & masks[other]) for one, other in pairs])

    def _held(self, masks: np.ndarray, held: np.ndarray, grown: object) -> np.ndarray:
        if isinstance(grown, tuple):
            one, other = grown
            return held & ~(masks[one] & masks[other])
        return held & masks[int(grown)]  # type: ignore[arg-type]

    def _counted(self, masks: np.ndarray, rows: int) -> np.ndarray:
        """How many places meet each candidate, per row: the number a pattern reads."""
        shaped = masks.reshape(masks.shape[0], rows, -1) if masks.ndim == 2 else masks.reshape(1, rows, -1)
        return shaped.sum(axis=2).astype(float)

    def _told_of(self, counted: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        """How much each candidate accounts for what is missed, as a correlation ignoring which way it points.

        Either direction is a term that has seen something; which way it points is the weight's business, the
        same reading `WentWithWinning` takes of a candidate rule."""
        spread = counted.std(axis=1)
        told = np.zeros(counted.shape[0])
        usable = spread > 0
        if not usable.any() or float(np.std(wanted)) == 0.0:
            return told
        middle = counted[usable] - counted[usable].mean(axis=1, keepdims=True)
        theirs = wanted - wanted.mean()
        told[usable] = np.abs(middle @ theirs) / (np.sqrt((middle**2).sum(axis=1)) * np.sqrt((theirs**2).sum()) + 1e-12)
        return told

    def _places(self, anchor: str, rows: Sequence[PositionRow]) -> tuple[tuple[object, ...], ...] | None:
        """Every place of the anchor, where every row has one of the same shape; None otherwise."""
        grids = [row.state.model(anchor) if row.state.has(anchor) else None for row in rows]
        if any(one is None or not isinstance(one, Grid) for one in grids):
            return None
        shapes = {one.shape for one in grids}  # type: ignore[union-attr]
        return grids[0].coordinates() if len(shapes) == 1 else None  # type: ignore[union-attr]

    def _offered(
        self,
        anchor: str,
        places: Sequence[tuple[object, ...]],
        rows: Sequence[PositionRow],
        vocabulary: Vocabulary,
    ) -> list[tuple[PatternCondition, np.ndarray]]:
        """Every condition that could go into a term, with where it holds: one truth per place per row.

        Conditions over the bases sharing the anchor's places, at every offset the vocabulary saw, against
        every value seen. **A base holding players' names gives `me` and `other` and never `white`**, so a
        term assembled from one side's positions is the same term read for the other — which is the whole of
        why a heuristic learned by one player is worth anything to the other.

        A condition true everywhere or nowhere is left out: it separates no place from any other, so it cannot
        be part of why one position is mispriced and another is not. That is `ClauseLearner`'s subtraction of
        what is read the same way of every case, done here as it is offered rather than afterwards."""
        anchor_places = vocabulary.indices_by_base.get(anchor)
        arity = len(next(iter(anchor_places))) if anchor_places else 0
        offsets = ((0,) * arity, *vocabulary.offsets_by_arity.get(arity, ()))
        offered: list[tuple[PatternCondition, np.ndarray]] = []
        for base in sorted(vocabulary.grids):
            if vocabulary.indices_by_base.get(base) != anchor_places:
                continue
            values = vocabulary.values_by_base.get(base, ())
            if any(isinstance(one, int | float) and not isinstance(one, bool) for one in values):
                continue
            tokens: list[tuple[str, object]] = [(repr(one), one) for one in values if one not in vocabulary.players]
            if any(one in vocabulary.players for one in values):
                tokens.extend(((ME, ME), (OTHER, OTHER)))
            read = [self._read(base, row, places) for row in rows]
            for steps in offsets:
                for rendered, wanted in tokens:
                    mask = np.concatenate([
                        self._holds(read[at], places, steps, wanted, rows[at].player, vocabulary)
                        for at in range(len(rows))
                    ])
                    if mask.any() and not mask.all():
                        offered.append((PatternCondition(base, steps, "==", rendered), mask))
        return offered

    def _read(self, base: str, row: PositionRow, places: Sequence[tuple[object, ...]]) -> dict:
        grid = row.state.model(base) if row.state.has(base) else None
        return {} if not isinstance(grid, Grid) else {where: grid.at(where) for where in grid.coordinates()}

    def _holds(
        self,
        read: dict,
        places: Sequence[tuple[object, ...]],
        steps: tuple[int, ...],
        wanted: object,
        player: str,
        vocabulary: Vocabulary,
    ) -> np.ndarray:
        """Where that condition holds in one row, one truth per place of the anchor."""
        held = wanted
        if wanted is ME:
            held = player
        elif wanted is OTHER:
            held = next((one for one in vocabulary.players if one != player), player)
        return np.array(
            [read.get(tuple(part + step for part, step in zip(where, steps, strict=True))) == held for where in places]
        )

    def _template(self, pattern: Pattern, vocabulary: Vocabulary) -> str:
        from openmind.inference.service.expression_generator import ExpressionGenerator

        return ExpressionGenerator().pattern_expression(pattern, vocabulary).template

    def _idioms(
        self,
        idioms: Sequence[tuple[str, str]],
        places: Sequence[tuple[object, ...]],
        rows: Sequence[PositionRow],
        says: object,
    ) -> list[tuple[PatternCondition, np.ndarray]]:
        """The named terms already earned, offered as single conditions alongside the cells.

        **This is what makes a long term reachable at all.** A fork is a dozen conditions over cells and two
        over idioms, and no search in this literature looks past four to six. So a term is never assembled
        long: what was found is named, comes back as one condition, and the next round is short again.

        `says` reads one idiom at one place of one row — what a view is for. Given none, there are no idioms
        and the assembly is over cells alone, which is where every vocabulary starts."""
        if not idioms or says is None:
            return []
        found: list[tuple[PatternCondition, np.ndarray]] = []
        for name, source in idioms:
            mask = np.concatenate([
                np.array([bool(says(row, source, where)) for where in places]) for row in rows
            ])
            if mask.any() and not mask.all():
                found.append((PatternCondition("", (), "", None, None, source, name), mask))
        return found

    def worded(self, pattern: Pattern) -> str:
        """What that term says, in words, so a thing with no name can still be named.

        **Described from its own structure, because there is nothing else honest to describe it from.** An
        idiom has to be called something before the next round can speak of it, and the only thing known about
        a term the moment it is assembled is what it reads. So the name is a reading of the conditions, and a
        term nobody could phrase comes out clumsy rather than wrong — which is the right way round.

        Offsets stay numbers. Which way is forward is not something a vocabulary of places knows, and a name
        that said "ahead" would be claiming a direction nobody deduced."""
        whole = self._said(pattern.conditions) or "anywhere"
        if pattern.absences:
            whole += ", and no " + ", nor ".join(self._said(group) for group in pattern.absences)
        return whole

    def _said(self, conditions: Sequence[PatternCondition]) -> str:
        """Conditions in words, those about one place said together: `a pawn of theirs at (-1, 0)`.

        Said together because that is what they mean. A kind and an owner are two conditions of one base each,
        and read apart they come out as "pawn there, theirs there", which is the same words in a worse order."""
        by_place: dict[tuple[int, ...], list[PatternCondition]] = {}
        named = [one.name for one in conditions if one.name]
        for one in conditions:
            if not one.name:
                by_place.setdefault(tuple(one.steps), []).append(one)
        for steps, held in by_place.items():
            words = [self._word(one) for one in held]
            where = "here" if not any(steps) else f"at {steps}"
            named.append(f"{' '.join(words)} {where}")
        return ", ".join(named)

    def _word(self, condition: PatternCondition) -> str:
        """One condition as a word: what it says a place holds, or whose it is."""
        value = str(condition.value)
        return {"me": "of mine", "other": "of theirs"}.get(value, value.strip("\'"))
