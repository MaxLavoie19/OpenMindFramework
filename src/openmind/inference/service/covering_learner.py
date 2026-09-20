import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openmind.inference.service.rule_deducer import Condition
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Covering:
    """One way an action can be legal: the conditions that hold together, and what they covered.

    `covers` is how many legal actions it accounts for and `wrongly` how many illegal ones it lets through — zero
    where it was learned to the end. A rule of a game is a way of being legal, not a filter: a knight's move and a
    bishop's move are two rules, and a move is legal where any of them covers it."""

    conditions: tuple[Condition, ...]
    covers: int
    wrongly: int

    @property
    def readable(self) -> str:
        return " and ".join(f"{reading} {relation} {value!r}" for reading, relation, value in self.conditions) or "anything"


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
        most: int = 20,
        least: int = 1,
        grow: float = 0.67,
    ) -> tuple[Covering, ...]:
        """The ways of being legal it found: at most `most` of them, each covering at least `least` legal actions.

        A rule that cannot be made to let nothing illegal through is kept all the same, with what it lets through
        counted: the readings may not be able to say what the game is doing, and saying so is better than saying
        nothing.

        `grow` is the share of the evidence a rule is grown from; the rest is what it is pruned against. Growing and
        pruning on the same actions leaves a rule holding whatever happened to separate them — the clock at forty,
        the game's own history — because dropping it would, on those very actions, let something illegal through.
        Only actions the rule was not grown on can tell an accident from a reason."""
        growing = examples[: int(len(examples) * grow)] or list(examples)
        pruning = examples[int(len(examples) * grow) :] or list(examples)
        illegal = [readings for readings, legal in growing if not legal]
        left = [readings for readings, legal in growing if legal]
        others = [readings for readings, legal in pruning if not legal]
        keeping = [readings for readings, legal in pruning if legal]
        found: list[Covering] = []
        while left and len(found) < most:
            rule = self._pruned(self._covering(left, illegal), [*left, *keeping], [*illegal, *others])
            covered = [readings for readings in left if self._matches(rule, readings)]
            if len(covered) < least:
                break
            wrongly = sum(1 for readings in illegal if self._matches(rule, readings))
            found.append(Covering(rule, len(covered), wrongly))
            left = [readings for readings in left if not self._matches(rule, readings)]
        logger.info(
            "Learned %d ways of being legal, covering %d of %d legal actions",
            len(found),
            sum(one.covers for one in found),
            sum(1 for _, legal in examples if legal),
        )
        return tuple(found)

    def covers(self, rules: Sequence[Covering], readings: Mapping[str, Value]) -> bool:
        """Whether any rule covers that action, which is what makes it legal."""
        return any(self._matches(rule.conditions, readings) for rule in rules)

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

    def _pruned(
        self,
        conditions: Sequence[Condition],
        left: Sequence[Mapping[str, Value]],
        illegal: Sequence[Mapping[str, Value]],
    ) -> tuple[Condition, ...]:
        """The rule with everything dropped that wasn't doing anything.

        A rule is grown by adding whatever separates, and what separates is as often an accident as a reason: the
        clock reading forty, the castling rights being gone, the position's own history. Each condition is taken out
        in turn and put back only where taking it out let an illegal action through. What is left is what the rule
        was really saying, and it is what carries to a position nothing here has seen."""
        kept = list(conditions)
        for condition in reversed(list(kept)):
            without = [held for held in kept if held != condition]
            if any(self._matches(without, readings) for readings in illegal):
                continue
            if not any(self._matches(without, readings) for readings in left):
                continue
            kept = without
        return tuple(kept)

    def _covering(
        self, left: Sequence[Mapping[str, Value]], illegal: Sequence[Mapping[str, Value]]
    ) -> tuple[Condition, ...]:
        """One rule: conditions added until no illegal action is covered, each chosen for how much of what it still
        wrongly covers it rules out while giving up as few legal actions as it can."""
        conditions: tuple[Condition, ...] = ()
        covering, wrongly = list(left), list(illegal)
        while wrongly:
            best, worth = None, 0.0
            for condition in self._questions(covering):
                kept = sum(1 for readings in covering if self._holds(readings, condition))
                if not kept:
                    continue
                ruled_out = sum(1 for readings in wrongly if not self._holds(readings, condition))
                if not ruled_out:
                    continue
                held = ruled_out * kept / len(covering)
                if held > worth:
                    best, worth = condition, held
            if best is None:
                break
            conditions = (*conditions, best)
            covering = [readings for readings in covering if self._holds(readings, best)]
            wrongly = [readings for readings in wrongly if self._holds(readings, best)]
        return conditions

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
