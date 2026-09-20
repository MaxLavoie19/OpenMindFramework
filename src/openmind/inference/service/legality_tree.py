import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from openmind.inference.service.rule_deducer import Condition
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: What a leaf says about the actions that reach it.
ALLOWED, REFUSED, UNKNOWN = "allowed", "refused", "unknown"


@dataclass(slots=True)
class Branch:
    """One question and where its answers lead, or a leaf and what it says.

    A leaf says allowed where every action reaching it was legal, refused where none was, and unknown where the
    readings couldn't tell them apart — which is not a failure but the case to go and look into."""

    says: str = UNKNOWN
    legal: int = 0
    illegal: int = 0
    condition: Condition | None = None
    yes: "Branch | None" = None
    no: "Branch | None" = None

    @property
    def leaf(self) -> bool:
        return self.condition is None


class LegalityTree:
    """What makes an action legal, learned as a tree of questions rather than as a heap of guesses.

    Asking what every legal action has in common gives a combinatorial pile: every reading against every value,
    within every case, most of them true by accident. A tree asks one question at a time — whichever single reading
    best separates the legal actions from the rest — and asks the next question only of what is left. That is one
    pass over the examples per level instead of a search over every hypothesis.

    What comes out reads back as rules: each path down to a leaf that refuses is a conjunction of conditions under
    which the game allows nothing, which is a constraint, and the rule-based system is those constraints. So the tree
    is how it is learned and the rules are what it becomes."""

    def fit(
        self,
        examples: Sequence[tuple[Mapping[str, Value], bool]],
        depth: int = 6,
        least: int = 2,
    ) -> Branch:
        """The tree that tells the legal actions from the rest, asking at most `depth` questions and never splitting
        fewer than `least` actions."""
        legal = sum(1 for _, allowed in examples if allowed)
        weight = (len(examples) - legal) / legal if legal else 1.0
        grown = self._grow(examples, depth, least, weight)
        logger.info(
            "Grew a tree of %d leaves over %d actions, %d of them legal",
            self._leaves(grown),
            len(examples),
            sum(1 for _, legal in examples if legal),
        )
        return grown

    def refuses(self, tree: Branch) -> tuple[tuple[Condition, ...], ...]:
        """Every path down to a leaf that refuses: the conditions under which the game allows nothing, which is what
        a constraint says. A path holds a condition where it was answered yes, and its opposite where no."""
        return tuple(self._paths(tree, (), REFUSED))

    def allows(self, tree: Branch) -> tuple[tuple[Condition, ...], ...]:
        """Every path down to a leaf that allows."""
        return tuple(self._paths(tree, (), ALLOWED))

    def unknown(self, tree: Branch) -> tuple[tuple[Condition, ...], ...]:
        """Every path down to a leaf where the readings couldn't tell legal from illegal: what to go and look at."""
        return tuple(self._paths(tree, (), UNKNOWN))

    def says(self, tree: Branch, readings: Mapping[str, Value]) -> str:
        """What the tree says about that action: allowed, refused, or unknown."""
        branch = tree
        while not branch.leaf:
            branch = branch.yes if self._holds(readings, branch.condition) else branch.no  # type: ignore[assignment,arg-type]
        return branch.says

    def _grow(
        self, examples: Sequence[tuple[Mapping[str, Value], bool]], depth: int, least: int, weight: float
    ) -> Branch:
        legal = sum(1 for _, allowed in examples if allowed)
        branch = Branch(self._says(legal, len(examples) - legal), legal, len(examples) - legal)
        if depth <= 0 or not legal or legal == len(examples) or len(examples) < least:
            return branch
        condition = self._best(examples, weight)
        if condition is None:
            return branch
        yes = [one for one in examples if self._holds(one[0], condition)]
        no = [one for one in examples if not self._holds(one[0], condition)]
        if not yes or not no:
            return branch
        branch.condition = condition
        branch.yes = self._grow(yes, depth - 1, least, weight)
        branch.no = self._grow(no, depth - 1, least, weight)
        return branch

    def _best(self, examples: Sequence[tuple[Mapping[str, Value], bool]], weight: float) -> Condition | None:
        """The one question that best separates the legal actions from the rest: the lowest weighed disorder of the
        two sides it makes."""
        best, lowest = None, self._disorder(examples, weight)
        for condition in self._questions(examples):
            yes = [one for one in examples if self._holds(one[0], condition)]
            if not yes or len(yes) == len(examples):
                continue
            no = [one for one in examples if not self._holds(one[0], condition)]
            weighed = (len(yes) * self._disorder(yes, weight) + len(no) * self._disorder(no, weight)) / len(examples)
            if weighed < lowest:
                best, lowest = condition, weighed
        return best

    def _questions(self, examples: Sequence[tuple[Mapping[str, Value], bool]]) -> list[Condition]:
        """What can be asked here: what each reading is, and where each number stands. Only the values the legal
        actions take are worth asking about — a question no legal action answers yes to separates nothing worth
        separating."""
        legal = [readings for readings, allowed in examples if allowed]
        if not legal:
            return []
        questions: list[Condition] = []
        for reading in sorted(legal[0]):
            values = {readings.get(reading) for readings in legal}
            numbers = sorted(
                value for value in values if isinstance(value, int | float) and not isinstance(value, bool)
            )
            if len(numbers) == len(values) and len(numbers) > 1:
                questions.extend((reading, "<=", number) for number in numbers[:-1])
                questions.extend((reading, ">=", number) for number in numbers[1:])
            questions.extend((reading, "==", value) for value in sorted(values, key=repr))
        return questions

    def _paths(self, branch: Branch, held: tuple[Condition, ...], says: str) -> list[tuple[Condition, ...]]:
        if branch.leaf:
            return [held] if branch.says == says else []
        condition = branch.condition
        assert condition is not None
        reading, relation, value = condition
        return [
            *self._paths(branch.yes, (*held, condition), says),  # type: ignore[arg-type]
            *self._paths(branch.no, (*held, (reading, f"not {relation}", value)), says),  # type: ignore[arg-type]
        ]

    def _says(self, legal: int, illegal: int) -> str:
        if legal and not illegal:
            return ALLOWED
        return REFUSED if illegal and not legal else UNKNOWN

    def _disorder(self, examples: Sequence[tuple[Mapping[str, Value], bool]], weight: float = 1.0) -> float:
        """How mixed those actions are: none where they are all legal or all not.

        The legal ones count for more, by however many times rarer they are. A game refuses almost everything — four
        thousand candidates to twenty legal moves — so a question that carves up what is refused looks like an
        excellent question and teaches nothing. Weighing the rare side up is what makes the tree ask what makes an
        action legal rather than what makes it one of the many that aren't."""
        legal = sum(1 for _, allowed in examples if allowed)
        illegal = len(examples) - legal
        weighed = legal * weight
        if not weighed or not illegal:
            return 0.0
        share = weighed / (weighed + illegal)
        return -(share * math.log2(share) + (1 - share) * math.log2(1 - share))

    def _leaves(self, branch: Branch) -> int:
        return 1 if branch.leaf else self._leaves(branch.yes) + self._leaves(branch.no)  # type: ignore[arg-type]

    def _holds(self, readings: Mapping[str, Value], condition: Condition) -> bool:
        reading, relation, value = condition
        held = readings.get(reading)
        if relation.startswith("not "):
            return not self._holds(readings, (reading, relation[4:], value))
        if relation == "==":
            return held == value
        if not isinstance(held, int | float) or isinstance(held, bool):
            return False
        return held <= value if relation == "<=" else held >= value  # type: ignore[operator]
