import logging
from collections.abc import Mapping, Sequence

from openmind.inference.model.fact import Fact
from openmind.inference.service.covering_learner import Covering
from openmind.inference.service.rule_reasoner import RuleReasoner
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: How far a thing's rules let it go, counted over the board's shape.
REACHES = "reaches"
#: Everything one thing's rules allow, another's allow too.
TAKES_IN = "takes in"
#: What a thing is worth to whoever holds it, so far as anything concluded says.
WORTH_AT_LEAST = "is worth at least"
#: A change one of a thing's actions can bring about, under the conditions the two rules share.
CAN_BRING_ABOUT = "can bring about"
#: What bringing that change about is worth, from what the thing it is brought about upon is worth.
THREATENS = "threatens"
#: How many things one action may put at stake at once, and so how many cannot all be answered in one turn.
AT_ONCE = "puts at stake at once"
#: That a player acts once before the others do. Not read off any rule: it is what the game does, told to the chain
#: as a fact of its own so that what rests on it says so.
ACTS_ONCE = "acts once in a turn"
#: That a thing can put more than one at stake at a time, so that they cannot all be got out of the way.
CANNOT_ALL_ESCAPE = "puts more at stake than can be answered"


class Inferrer:
    """Makes new facts out of the rules and the facts already had, until nothing further follows.

    Induction gives rules from what was seen, and deduction gives a position's value by playing it out. Neither
    makes anything new out of what is already known, and that is what inference is: two things held together give a
    third, which held against a fourth gives a fifth. A queen's rules take in a rook's, so a queen is worth at
    least what a rook is; a rook's rules reach fourteen squares, so a rook is worth at least fourteen; therefore a
    queen is worth at least fourteen — and the last of those was concluded from two conclusions, not from any rule
    and not from any board.

    Every fact carries what it rests on, so a chain can be read back to the rules it started from."""

    def __init__(self, reasoner: RuleReasoner | None = None) -> None:
        self._reasoner = reasoner or RuleReasoner()

    def chain(
        self,
        by: Mapping[Value, Sequence[Covering]],
        shape: tuple[int, ...],
        known: Sequence[Fact] = (),
        doing: Sequence[object] = (),
    ) -> tuple[Fact, ...]:
        """Everything that follows from those rules and those facts, chained until nothing further does.

        `by` is each thing's rules, by what the thing is — a piece kind, an action name, whatever the game has, and
        `doing` what its actions do. The first facts come from reading those; the rest come from holding facts
        against one another.

        `known` is what is already held, including facts about the game that no rule states — that a player acts
        once before the others do, which is what makes putting two things at stake at a time worth anything. It is
        given rather than read, and whatever rests on it says so."""
        facts = list(known)
        facts.extend(self._read(by, shape))
        facts.extend(self._brought(by, doing))
        facts.extend(self._at_once(by, shape))
        while True:
            further = [one for one in self._further(facts) if not self._held(facts, one)]
            if not further:
                break
            facts.extend(further)
            logger.info("Chained %d further facts", len(further))
        logger.info("Concluded %d facts from the rules of %d things", len(facts), len(by))
        return tuple(facts)

    def _read(self, by: Mapping[Value, Sequence[Covering]], shape: tuple[int, ...]) -> list[Fact]:
        """The facts that come from reading the rules: how far each thing reaches, and which takes in which."""
        found: list[Fact] = []
        for what, rules in by.items():
            reaches = self._reasoner.reaching(rules, shape)
            found.append(Fact(REACHES, (what,), float(reaches), tuple(one.readable for one in rules)))
        for what, mine in by.items():
            for other, theirs in by.items():
                if what != other and self._reasoner.within(mine, theirs):
                    found.append(Fact(TAKES_IN, (what, other), None, tuple(one.readable for one in mine)))
        return found

    def _brought(self, by: Mapping[Value, Sequence[Covering]], doing: Sequence[object]) -> list[Fact]:
        """What a thing's actions can bring about: a change whose conditions can hold together with a rule that
        allows the action.

        This is the step that reads what an action *does* rather than what it is. A rule says a move is allowed
        where the target holds another player's piece; a consequence says a move removes what stands at the target,
        under conditions of its own. Where those conditions can hold at once, the thing can bring that change about
        — which is a threat, said as a rule of the game and not as something spotted on a board.

        What is removed is whatever stands where the move lands, so the fact does not name it: it says this thing
        can remove, and what it can remove is anything that may stand there. Holding that against what each thing is
        worth gives what is threatened and for how much."""
        found: list[Fact] = []
        for what, rules in by.items():
            for consequence in doing:
                for when in getattr(consequence, "when", ()) or (Covering((), 0, 0),):
                    if any(self._reasoner.together(rule, when) for rule in rules):
                        found.append(
                            Fact(
                                CAN_BRING_ABOUT,
                                (what, getattr(consequence, "change", "?"), getattr(consequence, "model", "?")),
                                None,
                                tuple(one.readable for one in rules),
                            )
                        )
                        break
        return found

    def _at_once(self, by: Mapping[Value, Sequence[Covering]], shape: tuple[int, ...]) -> list[Fact]:
        """How many things a thing may have at stake at the same time.

        What a rule admits from a square it admits all at once — nothing is spent to have the rest — so a thing
        standing somewhere bears on everything its rule reaches from there. Where a player acts once in a turn, only
        one of those can be answered before the next, so anything above one cannot all be escaped."""
        return [
            Fact(AT_ONCE, (what,), float(self._reasoner.reaching(rules, shape)), tuple(one.readable for one in rules))
            for what, rules in by.items()
        ]

    def _further(self, facts: Sequence[Fact]) -> list[Fact]:
        """What follows from holding those facts against one another."""
        found: list[Fact] = []
        for one in facts:
            if one.kind == REACHES and one.held is not None:
                found.append(Fact(WORTH_AT_LEAST, one.about, one.held, (), (one,)))
        worth = {one.about[0]: one for one in facts if one.kind == WORTH_AT_LEAST and one.held is not None}
        for one in facts:
            if one.kind != TAKES_IN:
                continue
            held = worth.get(one.about[1])
            if held is not None and held.held is not None:
                found.append(Fact(WORTH_AT_LEAST, (one.about[0],), held.held, (), (one, held)))
        for one in facts:
            if one.kind != CAN_BRING_ABOUT or one.about[1] != "Removed":
                continue
            for what, held in worth.items():
                if held.held is None or what == one.about[0]:
                    continue
                found.append(Fact(THREATENS, (one.about[0], what), held.held, (), (one, held)))
        once = next((one for one in facts if one.kind == ACTS_ONCE), None)
        if once is not None:
            for one in facts:
                if one.kind == AT_ONCE and one.held is not None and one.held > 1:
                    found.append(Fact(CANNOT_ALL_ESCAPE, one.about, one.held, (), (one, once)))
        return found

    def _held(self, facts: Sequence[Fact], one: Fact) -> bool:
        """Whether that fact is already had, or a stronger one is: saying a thing is worth at least five adds
        nothing where it is already worth at least eight."""
        for held in facts:
            if held.kind != one.kind or held.about != one.about:
                continue
            if one.held is None or (held.held is not None and held.held >= one.held):
                return True
        return False
