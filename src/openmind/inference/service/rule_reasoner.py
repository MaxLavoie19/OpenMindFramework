import logging
from collections.abc import Mapping, Sequence

from openmind.inference.service.covering_learner import Covering
from openmind.inference.service.rule_deducer import Condition

logger = logging.getLogger(__name__)

#: How the readings that speak of the step between two squares are named, as `ActionReadings` names them for an
#: action whose two parameters are called source and target. A game naming them otherwise says so.
ABOUT = {
    "rows": "rows from source to target",
    "columns": "columns from source to target",
    "rows_apart": "rows apart, source and target",
    "columns_apart": "columns apart, source and target",
    "steps": "steps from source to target",
    "straight": "source and target share a row or a column",
    "diagonal": "source and target are on a diagonal",
}


class RuleReasoner:
    """Concludes things about a game from its rules, without looking at a position.

    Everything else OMF does to work a game out goes through positions: it plays them, gathers them, takes pieces
    off them and counts. That is measurement, and it can only ever say what was so in the positions it saw. A rule
    is a statement, and statements have consequences that hold in every position and can be had by reading them.

    The one drawn here is entailment: whether everything one rule allows, another allows too. It is decided by the
    conditions alone — a rule asking less of an action than another, and nothing the other does not ask, allows
    everything the other allows and more. From it follow orderings nothing needs a board to know: that what a queen
    may do includes what a rook may do, so a queen is worth at least a rook, in any position, always."""

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

    def reaching(self, rules: Sequence[Covering], shape: tuple[int, ...], about: Mapping[str, str] | None = None) -> float:
        """How far those rules reach: how many steps from a square to another they admit, over a board of that shape.

        A movement rule is conditions on how one square stands to another — so many rows apart, on a diagonal, one
        step — and what they admit can be counted by reading them, without a board and without playing. What the
        rules also ask about what stands where is no part of it: that says when a move is available, not how far the
        thing can go.

        What comes back is how many squares it reaches from a square, averaged over the squares there are. Counting
        the steps themselves counts both ways along a line, and from any one square only one of those is on the
        board — so a rook would come out twice what it reaches.

        This is what a piece is worth to whoever holds it, before anything else is known. It cannot be had by
        watching positions, because a rook hemmed in behind its own pieces reaches nothing this move while its rule
        reaches the whole file — and it is the rule that says what the piece is for."""
        named = dict(about or ABOUT)
        admitted = {
            (rows, columns)
            for rows in range(-(shape[0] - 1), shape[0])
            for columns in range(-(shape[1] - 1), shape[1])
            if (rows or columns) and any(self._admits(rule, rows, columns, named) for rule in rules)
        }
        squares = [(row, column) for row in range(shape[0]) for column in range(shape[1])]
        reached = sum(
            sum(
                1
                for rows, columns in admitted
                if 0 <= row + rows < shape[0] and 0 <= column + columns < shape[1]
            )
            for row, column in squares
        )
        return reached / len(squares) if squares else 0.0

    def _admits(self, rule: Covering, rows: int, columns: int, named: Mapping[str, str]) -> bool:
        """Whether that rule admits a step of so many rows and columns, by the conditions that speak of the step."""
        offered = self._offsets(rows, columns, named)
        for reading, relation, value in rule.conditions:
            if reading not in offered:
                continue
            held = offered[reading]
            if relation == "==" and held != value:
                return False
            if relation == "<=" and not (self._number(held) and self._number(value) and held <= value):  # type: ignore[operator]
                return False
            if relation == ">=" and not (self._number(held) and self._number(value) and held >= value):  # type: ignore[operator]
                return False
            if relation in ("== reading", "!= reading"):
                other = offered.get(str(value))
                if other is None:
                    continue
                if (held == other) != (relation == "== reading"):
                    return False
        return True

    def _offsets(self, rows: int, columns: int, named: Mapping[str, str]) -> dict[str, object]:
        """What a step of so many rows and columns reads as."""
        return {
            named["rows"]: rows,
            named["columns"]: columns,
            named["rows_apart"]: abs(rows),
            named["columns_apart"]: abs(columns),
            named["steps"]: max(abs(rows), abs(columns)),
            named["straight"]: (rows == 0) != (columns == 0),
            named["diagonal"]: rows != 0 and abs(rows) == abs(columns),
        }

    def _number(self, value: object) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)
