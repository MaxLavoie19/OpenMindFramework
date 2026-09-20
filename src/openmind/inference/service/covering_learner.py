import logging
from collections.abc import Callable, Mapping, Sequence
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
        within: Sequence[object] = (),
        positions: int = 2,
    ) -> tuple[Covering, ...]:
        """The ways of being legal it found: at most `most` of them, each covering at least `least` legal actions.

        A rule that cannot be made to let nothing illegal through is kept all the same, with what it lets through
        counted: the readings may not be able to say what the game is doing, and saying so is better than saying
        nothing.

        `grow` is the share of the evidence a rule is grown from; the rest is what it is pruned against. Growing and
        pruning on the same actions leaves a rule holding whatever happened to separate them — the clock at forty,
        the game's own history — because dropping it would, on those very actions, let something illegal through.
        Only actions the rule was not grown on can tell an accident from a reason.

        `within` says which position each action was read in. A reading that never changes within a position — the
        clock, the castling rights, the positions seen so far — can be compared to another reading and mean
        something, but compared to a value of its own it can only ever name the position it was read in. Such a
        comparison explains every action there and nothing anywhere else, so it is not offered.

        `positions` is how many positions a condition must hold in to be part of a rule. No rule of a game is true
        in one position only, so a condition supported by one is an accident however well it separates what it was
        found in — the square a pawn just passed, the clock reading forty. This is what the filter above was reaching
        for and missed: what matters is not whether a reading is fixed within a position, but whether a condition has
        support across them."""
        self._identifying = self._position_constant(examples, within)
        self._supported = self._support(examples, within, positions)
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
        """The rule with everything dropped that wasn't earning its place.

        A rule is grown by adding whatever separates, and what separates is as often an accident as a reason: the
        clock reading forty, the castling rights being gone, the position's own history. Conditions are dropped from
        the end while dropping one leaves the rule no worse — worth being how much of what it covers is legal
        against how much isn't, counted on actions it wasn't grown from.

        Insisting that dropping a condition let nothing illegal through drops nothing at all: against forty thousand
        candidates, loosening a rule by anything lets something through. What tells an accident from a reason is
        whether the rule is worth more without it, not whether it is perfect without it."""
        kept = list(conditions)
        worth = self._worth(kept, left, illegal)
        for condition in reversed(list(conditions)):
            without = [held for held in kept if held != condition]
            if not without:
                continue
            held = self._worth(without, left, illegal)
            if held >= worth:
                kept, worth = without, held
        return tuple(kept)

    def _worth(
        self,
        conditions: Sequence[Condition],
        left: Sequence[Mapping[str, Value]],
        illegal: Sequence[Mapping[str, Value]],
    ) -> float:
        """What a rule is worth: how much of what it covers is legal against how much isn't, from -1 where it covers
        only what the game refuses to 1 where it covers only what the game allows."""
        covers = sum(1 for readings in left if self._matches(conditions, readings))
        wrongly = sum(1 for readings in illegal if self._matches(conditions, readings))
        return (covers - wrongly) / (covers + wrongly) if covers or wrongly else -1.0

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
                if not self._worth_asking(condition):
                    continue
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

    def _position_constant(
        self, examples: Sequence[tuple[Mapping[str, Value], bool]], within: Sequence[object]
    ) -> frozenset[str]:
        """The readings that never change within a position: what can name a position rather than describe an
        action."""
        if len(within) != len(examples):
            return frozenset()
        seen: dict[object, dict[str, set[Value]]] = {}
        for (readings, _), where in zip(examples, within, strict=True):
            held = seen.setdefault(where, {})
            for reading, value in readings.items():
                held.setdefault(reading, set()).add(value)
        names = set(examples[0][0]) if examples else set()
        return frozenset(
            reading
            for reading in names
            if all(len(held.get(reading, ())) <= 1 for held in seen.values()) and len(seen) > 1
        )

    def _questions(self, covering: Sequence[Mapping[str, Value]]) -> list[Condition]:
        """What can be asked of the actions this rule still covers: what each reading is, where each number stands,
        and which readings are the same as one another or never are."""
        if not covering:
            return []
        names = sorted(covering[0])
        questions: list[Condition] = []
        for reading in names:
            if reading in getattr(self, "_identifying", frozenset()):
                continue
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
