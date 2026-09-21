import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from openmind.inference.service.condition_masks import ConditionMasks
from openmind.inference.service.rule_deducer import Condition
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Covering:
    """One way an action can be legal: the conditions that hold together, and what they covered.

    `covers` is how many legal actions it accounts for and `wrongly` how many illegal ones it lets through — zero
    where it was learned to the end. A rule of a game is a way of being legal, not a filter: a knight's move and a
    bishop's move are two rules, and a move is legal where any of them covers it.

    `excluding` is what this rule lists that the game refuses anyway, stated as rules of its own. A game is written
    down that way — how a piece moves, and separately what forbids the move — and the exclusions belong to the rule
    they were learned under rather than to the set, so what forbids a pawn's move can never reach a rook's.

    `under` is what specializes it: the rule is a principle, and an action it covers is covered where one of these
    covers it too. A set of rules that each repeat what they have in common says the common part once for every
    rule it appears in and nowhere as itself, so what is true of the game is never stated and cannot be reasoned
    with. Said once, with the ways it is specialized under it, it is a principle — and what forbids it is learned
    against everything it covers rather than against one specialization at a time."""

    conditions: tuple[Condition, ...]
    covers: int
    wrongly: int
    excluding: tuple["Covering", ...] = ()
    under: tuple["Covering", ...] = ()

    @property
    def readable(self) -> str:
        said = " and ".join(f"{reading} {relation} {value!r}" for reading, relation, value in self.conditions) or "anything"
        if self.under:
            said += ", in particular " + " or ".join(f"({one.readable})" for one in self.under)
        if self.excluding:
            said += ", except where " + " or where ".join(one.readable for one in self.excluding)
        return said


class CoveringLearner:
    """Learns what makes an action legal as a set of ways of being legal, one at a time.

    Being legal is not one thing. A knight's move is legal for reasons a pawn's is not, and a pawn has two ways of
    its own: it walks onto an empty square, or it takes across. A single set of conditions that every legal action
    satisfies cannot say that, which is why asking what they all have in common gives so little.

    So a rule is built to cover some of them rather than all: conditions are added to it, the most telling first,
    until it lets no illegal action through. What it covers is set aside and another rule is built for the rest,
    until nothing legal is left uncovered. What comes out is a rule set — legal where any rule covers it — which is
    what a rule-based system runs, each rule readable and each one droppable on its own."""

    def learn(
        self,
        examples: Sequence[tuple[Mapping[str, Value], bool]],
        least: int = 1,
        grow: float = 0.67,
        within: Sequence[object] = (),
        positions: int = 2,
        loosely: float = 0.0,
        overlapping: bool = False,
    ) -> tuple[Covering, ...]:
        """The ways of being legal it found, each covering at least `least` legal actions.

        It stops when every legal action is covered, or when no rule can be found that covers one nothing covers
        yet. There is no limit on how many rules it may take: how many a game needs is a fact about the game, not
        a number to be guessed beforehand, and a limit guessed too low truncates the game — the rules that would
        have said how a piece moves are never reached, and what is missing looks like something the learner could
        not learn. Every rule covers at least one action no other covers, so there are never more of them than
        there are legal actions to cover.

        A rule that cannot be made to let nothing illegal through is kept all the same, with what it lets through
        counted: the readings may not be able to say what the game is doing, and saying so is better than saying
        nothing. A rule with no conditions at all is another matter and is never kept: it says everything is
        allowed, which makes every other rule beside it pointless, and as an exclusion it forbids everything. Where
        nothing separates what is left, the honest answer is that nothing was learned about it.

        `grow` is the share of the evidence a rule is grown from; the rest is what it is pruned against. Growing and
        pruning on the same actions leaves a rule holding whatever happened to separate them — the clock at forty,
        the game's own history — because dropping it would, on those very actions, let something illegal through.
        Only actions the rule was not grown on can tell an accident from a reason.

        `within` says which position each action was read in, which is what `positions` counts support across.

        No reading is held back for fear of what it might be used to memorize. A reading that names the position it
        was read in is a reading a rule can be wrong with, and what answers that is evidence: a condition must hold
        across positions to be asked at all, a rule is pruned against actions it was not grown from, and what it is
        worth is measured on positions it has never seen. A filter that forbids a whole reading instead forbids
        every rule that needed it — the castling rights, whose turn it is — and hides the memorizing it was meant
        to prevent rather than showing it.

        `loosely` is how much of what it covers a rule may wrongly cover. A rule that must cover nothing illegal
        is driven narrow — every exception the game has becomes another condition, and what is left covers little.
        A loose rule is a draft: it says how a piece moves and leaves what forbids it to be said separately, which
        is how a game is written down in the first place. The drafts are sharpened by what rejects afterwards.

        `positions` is how many positions a condition must hold in to be part of a rule. No rule of a game is true
        in one position only, so a condition supported by one is an accident however well it separates what it was
        found in — the square a pawn just passed, the clock reading forty. This is what the filter above was reaching
        for and missed: what matters is not whether a reading is fixed within a position, but whether a condition has
        support across them.

        `overlapping` lets several rules cover the same action. Taking an action away once a rule covers it leaves
        every later rule to be grown from a residue, and a residue of a handful is separated by anything at all: the
        column a rook happened to move along becomes a condition and never comes out again, because there is nothing
        left to contradict it. Covered actions are weighed down instead — one an existing rule covers counts for
        1/(1 + how many cover it) — so what nothing covers still pulls hardest while a new rule may be grown right
        across the ones its predecessors have. Rules then overlap, which costs nothing to a set that is legal where
        any of them covers, and the rules that cover a family from several angles are the evidence a principle can
        be distilled from.

        Nothing here refutes the rules afterwards. Dropping a condition because it turns away actions the game
        allows was tried and is either destructive or inert: at a draft's looseness a rule covering a knight's two
        step shapes finds that asking for either turns the other away, drops both and comes out saying a knight may
        move anywhere; held to letting nothing illegal through, it never drops anything at all. What a refutation
        actually shows is that some legal actions want a rule of their own, which is a different thing to do with
        it, and `relaxed` and `challenged` are left for whoever does it."""
        self._supported = self._support(examples, within, positions)
        growing = examples[: int(len(examples) * grow)] or list(examples)
        pruning = examples[int(len(examples) * grow) :] or list(examples)
        illegal = [readings for readings, legal in growing if not legal]
        left = [readings for readings, legal in growing if legal]
        others = [readings for readings, legal in pruning if not legal]
        keeping = [readings for readings, legal in pruning if legal]
        masks = ConditionMasks([*illegal, *others])
        growing_wrongly = (1 << len(illegal)) - 1
        found: list[Covering] = []
        covered_by = [0] * len(left)
        while True:
            uncovered = [one for one, times in enumerate(covered_by) if not times]
            if not uncovered:
                break
            chosen = list(range(len(left))) if overlapping else uncovered
            covering = [left[one] for one in chosen]
            weights = [1.0 / (1.0 + covered_by[one]) for one in chosen]

            def grown(
                seed: Mapping[str, Value] | None = None,
                covering: Sequence[Mapping[str, Value]] = covering,
                weights: Sequence[float] = weights,
                chosen: Sequence[int] = chosen,
            ) -> tuple[tuple[Condition, ...], list[int]]:
                made = self._pruned(
                    self._covering(covering, masks, growing_wrongly, loosely, weights, seed),
                    [*covering, *keeping],
                    masks,
                    loosely,
                )
                return made, [one for one in chosen if self._matches(made, left[one])]

            rule, matched = grown()
            if overlapping and not any(not covered_by[one] for one in matched):
                rule, matched = grown(left[uncovered[0]])
            if not rule:
                logger.info("Nothing separates what is left from what the game refuses, so no rule is made of it")
                break
            if sum(1 for one in matched if not covered_by[one]) < least:
                break
            wrongly = self._wrongly(rule, masks, growing_wrongly)
            found.append(Covering(rule, len(matched), wrongly))
            for one in matched:
                covered_by[one] += 1
            logger.info(
                "Learned rule %d covering %d actions, %d wrongly, %d of %d left uncovered",
                len(found),
                len(matched),
                wrongly,
                sum(1 for times in covered_by if not times),
                len(left),
            )
        logger.info(
            "Learned %d ways of being legal, covering %d of %d legal actions, %d left uncovered",
            len(found),
            sum(1 for readings, legal in examples if legal and self.covers(tuple(found), readings)),
            sum(1 for _, legal in examples if legal),
            sum(1 for readings, legal in examples if legal and not self.covers(tuple(found), readings)),
        )
        return tuple(found)

    def distilled(
        self,
        rules: Sequence[Covering],
        examples: Sequence[tuple[Mapping[str, Value], bool]] = (),
        least: int = 2,
    ) -> tuple[Covering, ...]:
        """The rules restated as principles, with what specializes each one under it.

        Rules learned one at a time each carry the whole of what they need, so what the game is doing is spread
        across them and never said: that a piece must be the player's own appears in every rule and is a rule of
        none, and that a rook, a bishop and a queen all need an unobstructed way is stated three times as three
        unrelated facts. The conditions a family of rules shares are that family's principle, and what is left of
        each rule is how the principle is specialized. Distilled again under the principle, a family of families
        gives the levels a game is actually written in.

        A principle is only drawn where `least` rules share it — a core of one rule is that rule under another
        name. Where a rule holds nothing but the principle, the principle covers everything the family does on its
        own, and nothing is left to specialize.

        `examples` are what the principles are counted against, since what each one covers is no longer what the
        rules it was drawn from covered."""
        found = self._distilled(rules, least)
        return tuple(self._recounted(rule, examples) for rule in found) if examples else found

    def _distilled(self, rules: Sequence[Covering], least: int) -> tuple[Covering, ...]:
        left = list(rules)
        found: list[Covering] = []
        while left:
            shared: dict[Condition, int] = {}
            for rule in left:
                for condition in dict.fromkeys(rule.conditions):
                    shared[condition] = shared.get(condition, 0) + 1
            common = max(shared.values(), default=0)
            if common < least or common < 2:
                found.extend(left)
                break
            seed = next(condition for condition, times in shared.items() if times == common)
            family = [rule for rule in left if seed in rule.conditions]
            core = tuple(
                condition for condition in family[0].conditions if all(condition in rule.conditions for rule in family)
            )
            rest = [tuple(one for one in rule.conditions if one not in core) for rule in family]
            excluding = tuple(one for rule in family for one in rule.excluding)
            if any(not one for one in rest):
                found.append(Covering(core, 0, 0, excluding))
            else:
                under = self._distilled(
                    [
                        Covering(one, rule.covers, rule.wrongly, rule.excluding, rule.under)
                        for one, rule in zip(rest, family, strict=True)
                    ],
                    least,
                )
                found.append(Covering(core, 0, 0, (), under))
            left = [rule for rule in left if seed not in rule.conditions]
        return tuple(found)

    def _recounted(
        self,
        rule: Covering,
        examples: Sequence[tuple[Mapping[str, Value], bool]],
        inherited: tuple[Condition, ...] = (),
    ) -> Covering:
        """The rule with what it covers counted again, itself and everything under it.

        A specialization is counted with the conditions it inherits, since those are the only actions it is ever
        asked about: counted on its own it answers for actions its principle already turned away, and says it is
        wrong about thousands of them."""
        whole = (*inherited, *rule.conditions)
        stated = replace(rule, under=tuple(self._recounted(one, examples, whole) for one in rule.under))
        counting = replace(stated, conditions=whole)
        covers = sum(1 for readings, legal in examples if legal and self._covered(counting, readings, None))
        wrongly = sum(1 for readings, legal in examples if not legal and self._covered(counting, readings, None))
        return replace(stated, covers=covers, wrongly=wrongly)

    def excluding(
        self,
        rules: Sequence[Covering],
        examples: Sequence[tuple[Mapping[str, Value], bool]],
        within: Sequence[object] = (),
        positions: int = 2,
        grow: float = 0.67,
    ) -> tuple[Covering, ...]:
        """The rules again, each carrying what excludes the illegal actions it covers.

        A rule that lists how a piece moves lists moves the game refuses anyway, and what refuses them is a rule in
        its own right — the way is blocked, the king is left attacked. It is learned from the actions that rule
        covers and from no others: a rule's exclusions answer for the actions it is responsible for, so the reasons
        offered are the reasons that apply, and there are few enough of them to have something in common. Learning
        what forbids across the whole set instead puts a pawn's exceptions and a rook's in one pile, where what
        separates them is a coincidence.

        They are learned to the end, never loosely. A rule that lists too much is answered by an exclusion, but an
        exclusion that rejects too much loses a legal action and nothing gives it back.

        Where a rule is a principle, what forbids is learned at the principle first, against everything it covers,
        and only what is still wrongly covered goes to the specializations under it. A prohibition of the game holds
        of every piece it applies to, so it belongs where it can be seen — learned once at the level that covers
        them all, on all the evidence there is for it, rather than found again under each piece from a few actions
        each and stated as several unrelated exceptions.

        Every reading is offered at once, readings of the move and of what it leads to alike. Learning what a
        reading of the move can say first, and only then letting the rest be used, was tried and is worse: it finds
        fewer legal actions and uses no reading of the outcome at all. The two kinds of mistake do not compete for
        the learner's attention in the way that reasoning supposed.

        `examples` are read with what each action leads to, since that is what an exclusion speaks of."""
        mine = list(zip(examples, within or [None] * len(examples), strict=True))
        return tuple(self._excluding(rule, mine, bool(within), positions, grow) for rule in rules)

    def _excluding(
        self,
        rule: Covering,
        examples: Sequence[tuple[tuple[Mapping[str, Value], bool], object]],
        placed: bool,
        positions: int,
        grow: float,
        allowing: bool = True,
    ) -> Covering:
        """The rule with what answers its mistakes hung under it, and what answers those in turn.

        `allowing` says what this rule does to the actions it covers. A rule that allows is mistaken about the
        illegal ones it covers, and what answers that is a rule that rejects them. A rule that rejects is mistaken
        about the legal ones it covers, and what answers that is a rule that allows them again — an exception to
        the prohibition, which is how a game says en passant. So the same learning runs at every level with the
        sides swapped, and it keeps going while each level still has something to answer for and can find fewer
        actions to say it about than the level above."""
        covered = [one for one in examples if self._matches(rule.conditions, one[0][0])]
        mistaken = [(readings, legal != allowing) for (readings, legal), _ in covered]
        found: tuple[Covering, ...] = ()
        if any(wrong for _, wrong in mistaken):
            found = self.learn(
                mistaken,
                within=[where for _, where in covered] if placed else (),
                positions=positions,
                grow=grow,
            )
            logger.info(
                "A rule %s %d actions is wrong about %d of them, answered by %d rules",
                "allowing" if allowing else "rejecting",
                len(covered),
                sum(1 for _, wrong in mistaken if wrong),
                len(found),
            )
        wrong = sum(1 for _, one in mistaken if one)
        answering = tuple(self._answering(one, covered, placed, positions, grow, allowing, wrong) for one in found)
        left = [one for one in covered if not any(self._covered(each, one[0][0], None) for each in answering)]
        under = tuple(self._excluding(one, left, placed, positions, grow, allowing) for one in rule.under)
        return replace(rule, excluding=answering, under=under)

    def _answering(
        self,
        rule: Covering,
        examples: Sequence[tuple[tuple[Mapping[str, Value], bool], object]],
        placed: bool,
        positions: int,
        grow: float,
        allowing: bool,
        above: int,
    ) -> Covering:
        """That rule taken further, where taking it further is progress.

        A level is answered only where it is wrong about fewer actions than the level above it was. What is left to
        answer for then strictly falls and cannot fall below nothing, so the alternation always ends — where asking
        it to go on until it is right could never end on evidence the readings are unable to separate. A level that
        is wrong about as much as its parent has said nothing new, and saying it again in another form will not
        help."""
        wrong = sum(
            1 for (readings, legal), _ in examples if self._matches(rule.conditions, readings) and legal == allowing
        )
        if not wrong or wrong >= above:
            return rule
        return self._excluding(rule, examples, placed, positions, grow, not allowing)

    def challenging(self, rules: Sequence[Covering]) -> tuple[tuple[Covering, Condition], ...]:
        """Every rule paired with each of its own conditions: what to look for an exception to.

        A condition is only worth what the actions it turns away are worth. One that turns away nothing the game
        allows is a reason; one that turns away something is a rule stated too narrowly, and the action it turned
        away is the edge case that says so. What the pairs ask for is an action meeting every other condition of
        the rule and failing this one — so the rule that comes back carries the conditions it inherits from the
        principles above it, since a specialization is only ever asked about actions those already cover.

        Conditions inherited from a principle are challenged at the principle, where everything they apply to can
        be seen, and exclusions are not challenged here: they answer for what a rule wrongly covers, and what
        refutes one is that the game allowed the action after all, which is the same question asked of the rule
        above it."""
        return self._challenging(rules, ())

    def _challenging(
        self, rules: Sequence[Covering], inherited: tuple[Condition, ...]
    ) -> tuple[tuple[Covering, Condition], ...]:
        asking: list[tuple[Covering, Condition]] = []
        for rule in rules:
            whole = replace(rule, conditions=(*inherited, *rule.conditions))
            asking.extend((whole, condition) for condition in rule.conditions)
            asking.extend(self._challenging(rule.under, whole.conditions))
        return tuple(asking)

    def challenged(
        self, rules: Sequence[Covering], examples: Sequence[tuple[Mapping[str, Value], bool]]
    ) -> dict[tuple[Condition, bool], int]:
        """What each condition of each rule costs, counted over the actions it was asked about.

        Against a condition, `(condition, False)`: the legal actions that meet every other condition of its rule
        and fail this one. Each is a way of being legal the rule cannot state, and where there are any, the
        condition is wrong as it stands — it is either to be dropped, or to be split into the cases it was standing
        in for.

        For it, `(condition, True)`: the illegal actions its rule covers with that condition held. Those are what
        the condition failed to turn away, and they say the rule is loose somewhere else."""
        counted: dict[tuple[Condition, bool], int] = {}
        for rule, condition in self.challenging(rules):
            others = tuple(one for one in rule.conditions if one != condition)
            for readings, legal in examples:
                if legal and not self._holds(readings, condition) and self._matches(others, readings):
                    counted[(condition, False)] = counted.get((condition, False), 0) + 1
                elif not legal and self._matches(rule.conditions, readings):
                    counted[(condition, True)] = counted.get((condition, True), 0) + 1
        return dict(sorted(counted.items(), key=lambda pair: -pair[1]))

    def relaxed(
        self,
        rules: Sequence[Covering],
        examples: Sequence[tuple[Mapping[str, Value], bool]],
        loosely: float = 0.0,
    ) -> tuple[Covering, ...]:
        """The rules with every condition dropped that turns away actions the game allows.

        A condition earns its place by ruling something out. One that also rules *in* nothing — that turns away
        legal actions meeting every other condition of its rule — is not a reason, it is the shape of the evidence
        the rule happened to be grown from: the half of the diagonals the bishops in those positions went along,
        the direction the rooks happened to move. Sequential covering puts such conditions in and nothing ever takes
        them out, because on the actions the rule was grown from they were never contradicted.

        A condition is dropped where doing so recovers legal actions and lets through no more than `loosely` of
        what the rule covers. Dropping is repeated until nothing more can be: widening a rule exposes the next
        condition to actions it was never asked about.

        `loosely` is nothing to do with how loosely the rule was grown, and defaults to letting nothing illegal
        through. Looseness belongs to growing — stop adding conditions before the rule is perfect — and a rule
        refuted as loosely as it was grown destroys itself: one covering a knight's two step shapes finds that
        asking for either turns the other away, drops both, and comes out saying a knight may move anywhere."""
        legal = [readings for readings, one in examples if one]
        masks = ConditionMasks([readings for readings, one in examples if not one])
        found: list[Covering] = []
        for number, rule in enumerate(rules):
            kept, dropped = list(rule.conditions), 0
            while True:
                letting = self._droppable(kept, legal, masks, loosely)
                if letting is None:
                    break
                kept.remove(letting)
                dropped += 1
                logger.info("Dropped %r: it turns away actions the game allows", letting)
            logger.info(
                "Refuted rule %d of %d, dropping %d of %d conditions",
                number + 1,
                len(rules),
                dropped,
                len(rule.conditions),
            )
            found.append(replace(rule, conditions=tuple(kept)))
        return tuple(found)

    def _droppable(
        self,
        conditions: Sequence[Condition],
        legal: Sequence[Mapping[str, Value]],
        masks: ConditionMasks,
        loosely: float,
    ) -> Condition | None:
        """The condition worth dropping first: the one turning away the most the game allows."""
        best, worth = None, 0
        for condition in conditions:
            others = tuple(one for one in conditions if one != condition)
            covered = [readings for readings in legal if self._matches(others, readings)]
            turned = sum(1 for readings in covered if not self._holds(readings, condition))
            if turned <= worth:
                continue
            covers = len(covered)
            wrongly = self._wrongly(others, masks, masks.everything)
            if wrongly <= loosely * max(covers, 1):
                best, worth = condition, turned
        return best

    def uncovered(
        self, rules: Sequence[Covering], examples: Sequence[tuple[Mapping[str, Value], bool]], by: str
    ) -> dict[Value, int]:
        """The legal actions no rule covers, counted by what that reading says of them.

        What a learner has not managed to account for says where to look next: where the actions it cannot cover are
        mostly of one kind, what is missing is evidence of that kind, and the way to get it is to build positions
        that have it. A learner that only ever sees what a game happens to offer learns what the game happens to
        offer."""
        missing: dict[Value, int] = {}
        for readings, legal in examples:
            if legal and not self.covers(rules, readings):
                held = readings.get(by)
                missing[held] = missing.get(held, 0) + 1
        return dict(sorted(missing.items(), key=lambda pair: -pair[1]))

    def covers(
        self,
        rules: Sequence[Covering],
        readings: Mapping[str, Value],
        after: Callable[[], Mapping[str, Value] | None] | None = None,
    ) -> bool:
        """Whether any rule covers that action and none of that rule's own exclusions rejects it, which is what
        makes it legal.

        `after` gives what the action leads to, read as readings, and is asked for only once a rule has covered the
        action and that rule has something to exclude. An outcome is what a prediction costs, and a candidate no
        rule covers needs none. Where it cannot be told what the action leads to, the exclusions go unchecked and
        the rule covers: what forbids the move is a claim about a position, and there is no position to look at."""
        asked: list[Mapping[str, Value] | None] = []

        def outcome() -> Mapping[str, Value] | None:
            if after is None:
                return readings
            if not asked:
                asked.append(after())
            return asked[0]

        return any(self._covered(rule, readings, outcome) for rule in rules)

    def _covered(
        self,
        rule: Covering,
        readings: Mapping[str, Value],
        outcome: Callable[[], Mapping[str, Value] | None] | None,
    ) -> bool:
        """Whether that rule covers the action: its conditions hold, something under it covers the action where it
        is a principle, and nothing it excludes rejects the action.

        What it excludes is asked last on purpose. A principle covers a great many actions and what excludes it
        speaks of the position the action leads to, so asking it first has a position predicted and read for very
        nearly every candidate there is, most of which nothing under the principle covers anyway. Asked after, a
        prediction is only ever paid for by an action the rules actually allow."""
        if not self._matches(rule.conditions, readings):
            return False
        if rule.under and not any(self._covered(one, readings, outcome) for one in rule.under):
            return False
        if rule.excluding:
            after = readings if outcome is None else outcome()
            if after is not None and any(self._covered(one, after, outcome) for one in rule.excluding):
                return False
        return True

    def scored(
        self, rules: Sequence[Covering], examples: Sequence[tuple[Mapping[str, Value], bool]]
    ) -> tuple[int, int, int]:
        """What the rules are worth as a generator: legal actions covered, legal actions missed, illegal actions
        wrongly covered."""
        found = missed = wrong = 0
        for readings, legal in examples:
            if self.covers(rules, readings):
                found, wrong = (found + 1, wrong) if legal else (found, wrong + 1)
            elif legal:
                missed += 1
        return found, missed, wrong

    def _wrongly(self, conditions: Sequence[Condition], masks: ConditionMasks, within: int) -> int:
        """How many of those actions the rule covers."""
        for condition in conditions:
            within &= masks.holding(condition)
            if not within:
                break
        return within.bit_count()

    def _pruned(
        self,
        conditions: Sequence[Condition],
        left: Sequence[Mapping[str, Value]],
        masks: ConditionMasks,
        loosely: float = 0.0,
    ) -> tuple[Condition, ...]:
        """The rule with everything dropped that wasn't earning its place.

        A rule is grown by adding whatever separates, and what separates is as often an accident as a reason: the
        clock reading forty, the castling rights being gone, the position's own history. Conditions are dropped from
        the end while dropping one leaves the rule no worse — worth being how much of what it covers is legal
        against how much isn't, counted on actions it wasn't grown from.

        Insisting that dropping a condition let nothing illegal through drops nothing at all: against forty thousand
        candidates, loosening a rule by anything lets something through. What tells an accident from a reason is
        whether the rule is worth more without it, not whether it is perfect without it."""
        kept = list(conditions)
        worth = self._worth(kept, left, masks)
        for condition in reversed(list(conditions)):
            without = [held for held in kept if held != condition]
            if not without:
                continue
            held = self._worth(without, left, masks)
            if held >= worth:
                kept, worth = without, held
        return tuple(kept)

    def _worth(
        self,
        conditions: Sequence[Condition],
        left: Sequence[Mapping[str, Value]],
        masks: ConditionMasks,
    ) -> float:
        """What a rule is worth: how much of what it covers is legal against how much isn't, from -1 where it covers
        only what the game refuses to 1 where it covers only what the game allows."""
        covers = sum(1 for readings in left if self._matches(conditions, readings))
        wrongly = self._wrongly(conditions, masks, masks.everything)
        return (covers - wrongly) / (covers + wrongly) if covers or wrongly else -1.0

    def _covering(
        self,
        left: Sequence[Mapping[str, Value]],
        masks: ConditionMasks,
        wrongly: int,
        loosely: float = 0.0,
        weights: Sequence[float] = (),
        seed: Mapping[str, Value] | None = None,
    ) -> tuple[Condition, ...]:
        """One rule: conditions added until it covers little enough of what the game refuses, each chosen for how
        much of what it still wrongly covers it rules out while giving up as few legal actions as it can.

        `weights` is how much each legal action is still worth covering — one already covered is worth less, so the
        rule is drawn towards what nothing accounts for yet without being forbidden what is already accounted for.
        Weighing them all the same is sequential covering, which is what the residue gave.

        `seed` is one action the rule must cover, and only conditions that hold of it are offered. Weighing alone
        does not get a rule to an action nothing accounts for: a family of three already covered still outweighs a
        single one that is not, so the search returns the rule it already has and the odd action is never reached.
        Growing from an action outwards settles it — the rule is guaranteed to cover its seed, so there is always
        progress, and it is scored against every legal action rather than the residue, so it grows as wide as the
        evidence allows. An action left over stops forcing a rule of its own and becomes one more case of a rule
        that already accounts for others.

        It is asked for only once growing without it has covered nothing new. Seeding every rule spends the rules
        there are on whichever family the leftover action belongs to, in the order the actions happen to come in,
        and the large families that no rule has reached yet are never asked about at all."""
        conditions: tuple[Condition, ...] = ()
        weighed = list(zip(left, weights or [1.0] * len(left), strict=True))
        worth_covering = sum(weight for _, weight in weighed)
        standing = wrongly.bit_count()
        while standing and standing > loosely * max(worth_covering, 1.0):
            best, worth, narrowed = None, 0.0, wrongly
            for condition in self._questions([readings for readings, _ in weighed]):
                if not self._worth_asking(condition):
                    continue
                if seed is not None and not self._holds(seed, condition):
                    continue
                kept = sum(weight for readings, weight in weighed if self._holds(readings, condition))
                if not kept:
                    continue
                holding = wrongly & masks.holding(condition)
                ruled_out = standing - holding.bit_count()
                if not ruled_out:
                    continue
                held = ruled_out * kept / worth_covering
                if held > worth:
                    best, worth, narrowed = condition, held, holding
            if best is None:
                break
            conditions = (*conditions, best)
            weighed = [(readings, weight) for readings, weight in weighed if self._holds(readings, best)]
            wrongly = narrowed
            standing = wrongly.bit_count()
            worth_covering = sum(weight for _, weight in weighed)
            if not worth_covering:
                break
        return conditions

    def _support(
        self,
        examples: Sequence[tuple[Mapping[str, Value], bool]],
        within: Sequence[object],
        least: int,
    ) -> Callable[[Condition], bool] | None:
        """A test for whether a condition holds of legal actions in enough positions to be a rule rather than an
        accident. None where nothing says which position an action was read in."""
        if len(within) != len(examples) or least < 2:
            return None
        legal = [(readings, where) for (readings, allowed), where in zip(examples, within, strict=True) if allowed]

        def supported(condition: Condition) -> bool:
            seen: set[object] = set()
            for readings, where in legal:
                if where not in seen and self._holds(readings, condition):
                    seen.add(where)
                    if len(seen) >= least:
                        return True
            return False

        return supported

    def _questions(self, covering: Sequence[Mapping[str, Value]]) -> list[Condition]:
        """What can be asked of the actions this rule still covers: what each reading is, where each number stands,
        and which readings are the same as one another or never are."""
        if not covering:
            return []
        names = sorted(covering[0])
        questions: list[Condition] = []
        for reading in names:
            values = {readings.get(reading) for readings in covering}
            questions.extend((reading, "==", value) for value in sorted(values, key=repr))
            numbers = sorted(
                value for value in values if isinstance(value, int | float) and not isinstance(value, bool)
            )
            if len(numbers) == len(values) and len(numbers) > 1:
                questions.extend((reading, "<=", number) for number in numbers[:-1])
                questions.extend((reading, ">=", number) for number in numbers[1:])
        for number, first in enumerate(names):
            for second in names[number + 1 :]:
                alike = all(self._alike(readings.get(first), readings.get(second)) for readings in covering)
                if not alike:
                    continue
                same = [readings.get(first) == readings.get(second) for readings in covering]
                if all(same):
                    questions.append((first, "== reading", second))
                elif not any(same):
                    questions.append((first, "!= reading", second))
        return questions

    def _worth_asking(self, condition: Condition) -> bool:
        """Whether a condition has enough behind it to be part of a rule."""
        supported = getattr(self, "_supported", None)
        return supported is None or supported(condition)

    def _matches(self, conditions: Sequence[Condition], readings: Mapping[str, Value]) -> bool:
        return all(self._holds(readings, condition) for condition in conditions)

    def _alike(self, one: Value, other: Value) -> bool:
        return isinstance(one, bool) == isinstance(other, bool) and (
            isinstance(one, int | float) == isinstance(other, int | float)
        )

    def _holds(self, readings: Mapping[str, Value], condition: Condition) -> bool:
        reading, relation, value = condition
        held = readings.get(reading)
        if relation in ("== reading", "!= reading"):
            other = readings.get(str(value))
            if not self._alike(held, other):
                return False
            return held == other if relation == "== reading" else held != other
        if relation == "==":
            return held == value
        if not isinstance(held, int | float) or isinstance(held, bool):
            return False
        return held <= value if relation == "<=" else held >= value  # type: ignore[operator]
