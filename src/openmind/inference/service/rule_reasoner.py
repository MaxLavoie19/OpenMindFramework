import logging
from collections.abc import Sequence

from openmind.inference.service.covering_learner import Covering
from openmind.inference.service.rule_deducer import Condition

logger = logging.getLogger(__name__)

class RuleReasoner:
    """Concludes things about a game from its rules, without looking at a position.

    Everything else OMF does to work a game out goes through positions: it plays them, gathers them, takes pieces
    off them and counts. That is measurement, and it can only ever say what was so in the positions it saw. A rule
    is a statement, and statements have consequences that hold in every position and can be had by reading them.

    The one drawn here is entailment: whether everything one rule allows, another allows too. It is decided by the
    conditions alone — a rule asking less of an action than another, and nothing the other does not ask, allows
    everything the other allows and more. From it follows an ordering nothing needs a board to know: that what one
    rule allows takes in what another allows, in any position, always.

    **What that ordering is not is a worth.** Allowing more is not being worth more — that step is a theory about
    what wins, and taking it here would be valuing a thing by what its rules admit and calling the answer a
    discovery, which is the one thing OMF must not do. What a thing is worth is what the games say it is worth."""

    def entails(self, one: Covering, other: Covering) -> bool:
        """Whether everything `other` allows, `one` allows too: `one` asks no more than `other` asks.

        Read the other way round, `other` is the narrower rule. A rule asks for conditions to hold together, so
        asking for fewer of them, or for weaker ones, allows more."""
        return all(any(self.implies(held, asked) for held in other.conditions) for asked in one.conditions)

    def implies(self, held: Condition, asked: Condition) -> bool:
        """Whether a condition holding means another one does.

        Only what can be decided from the two conditions themselves: the same question answered the same way, or a
        bound that is tighter than the one asked for. Two conditions about different readings say nothing about one
        another, however they relate on a board."""
        reading, relation, value = held
        wanted, wanted_relation, wanted_value = asked
        if reading != wanted:
            return False
        if relation == wanted_relation and value == wanted_value:
            return True
        if not self._number(value) or not self._number(wanted_value):
            return False
        if relation == "==":
            if wanted_relation == "<=":
                return value <= wanted_value  # type: ignore[operator]
            if wanted_relation == ">=":
                return value >= wanted_value  # type: ignore[operator]
            return False
        if relation != wanted_relation:
            return False
        if relation == "<=":
            return value <= wanted_value  # type: ignore[operator]
        return value >= wanted_value  # type: ignore[operator]

    def within(self, rules: Sequence[Covering], others: Sequence[Covering]) -> bool:
        """Whether everything those rules allow, these allow too.

        A set of rules allows an action where any of them covers it, so it takes in another set where every rule of
        that set is entailed by one of these."""
        return all(any(self.entails(one, other) for one in rules) for other in others)

    def ordered(self, by: dict[object, Sequence[Covering]]) -> tuple[tuple[object, object], ...]:
        """Which of those things affords at least what another does, each pair concluded from their rules alone.

        What comes back is every pair where the first takes in the second and the second does not take in the
        first — an ordering of what a game's pieces are for, had before a single position is looked at."""
        found = []
        for one, mine in by.items():
            for other, theirs in by.items():
                if one == other:
                    continue
                if self.within(mine, theirs) and not self.within(theirs, mine):
                    found.append((one, other))
        logger.info("Concluded %d orderings from %d sets of rules, with no position looked at", len(found), len(by))
        return tuple(found)

    def together(self, one: Covering, other: Covering) -> bool:
        """Whether two rules' conditions can hold of the same action.

        Two rules are compatible unless they ask contradictory things of the same reading: the same reading equal to
        two different values, or bounds that cannot both hold. Anything else is left open, since a reading not
        spoken of by one rule is not denied by it."""
        for held in one.conditions:
            for asked in other.conditions:
                if held[0] != asked[0] or held == asked:
                    continue
                if self._against(held, asked):
                    return False
        return True

    def _against(self, held: Condition, asked: Condition) -> bool:
        """Whether two conditions about the same reading cannot both hold."""
        _, relation, value = held
        _, wanted_relation, wanted_value = asked
        if relation == "==" and wanted_relation == "==":
            return value != wanted_value
        if not self._number(value) or not self._number(wanted_value):
            return False
        if relation == "==" and wanted_relation == "<=":
            return value > wanted_value  # type: ignore[operator]
        if relation == "==" and wanted_relation == ">=":
            return value < wanted_value  # type: ignore[operator]
        if relation == "<=" and wanted_relation == ">=":
            return value < wanted_value  # type: ignore[operator]
        if relation == ">=" and wanted_relation == "<=":
            return value > wanted_value  # type: ignore[operator]
        return False

    def _number(self, value: object) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)
