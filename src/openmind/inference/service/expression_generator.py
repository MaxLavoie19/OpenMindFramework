import itertools
from collections.abc import Iterable, Sequence

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import (
    AGGREGATE_INDEX,
    AGGREGATE_OTHER_INDEX,
    AGGREGATES,
    BEST,
    BODY_OPERATIONS,
    COMBINATIONS,
    COUNT,
    HERE,
    SUM,
    UNARY,
    LOOK_AHEAD_VARIABLE,
    ME,
    OTHER,
    OUTSIDE,
    PATTERN_INDEX,
    PATTERN_RELATIONS,
    THRESHOLDS,
    VIEW,
    WORST,
)
from openmind.inference.model.aggregate import Aggregate
from openmind.inference.model.expression import Expression
from openmind.inference.model.pattern import Pattern
from openmind.inference.model.pattern_condition import PatternCondition
from openmind.inference.model.vocabulary import Vocabulary
from openmind.rule.model.python_rule import PythonRule
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State
from openmind.world.model.value import Value


class ExpressionGenerator:
    """Generates expressions for any domain from the positions of its rows and its players only; nothing here knows a
    game. A player's name, as a value or as an index, is written relative to the player valued: `me` or `other`.

    Values are either names (a mark, a piece, a player, None) or numbers (a position, a clock, a count); a variable or
    base is numeric when every value seen is a number, and a number is never compared for equality with a value seen.

    - Leaves: a numeric variable as it is (`{view}.x[me]`); every other variable at every value seen
      (`{view}.cell[2, 2] == me`), the player to act also absolutely (`{view}.turn == 'X'`); for every indexed base of
      names and value, how many indices hold it, a pattern of one condition; for every numeric indexed base, the sum,
      lowest and highest of its values, aggregates of one reading; and how many actions each player could take.
    - A pattern grows by one condition, for whole-number indices: any base read at the pattern's index shifted by any
      offset seen between two indices (the same index included), equal or not equal to any name its base takes, to
      OUTSIDE, or to the variable another condition reads.
    - An aggregate's body grows by an operation (`+ - * / abs >= <= == and or`) with a reading of any base sharing its
      indices, at `i`, or at `j` once it reads pairs; a body at `i` also turns into a comparison of itself at `i` and at
      `j`, over every pair of different indices. Each body is summed, counted, and taken at its lowest and highest.
    - Thresholds: an expression at least, or at most, each value it takes; and its absolute value.
    - Combinations of two expressions: `+`, `-`, `*`, `/ max(1, ·)`, `max`, `min`, `>=`, `==`.
    - Look-aheads, for `me` and for `other`: the best, the worst and the count of the expression after an action, and
      the best and worst change of the expression, and how many actions raise or lower it."""

    def __init__(self, variable_name_mapper: VariableNameMapper) -> None:
        self._variable_name_mapper = variable_name_mapper

    def vocabulary(self, domain: Domain, states: Iterable[State]) -> Vocabulary:
        values_by_variable: dict[str, dict[Value, None]] = {}
        for state in states:
            for name, value in state.variables:
                values_by_variable.setdefault(name, {})[value] = None
        values_by_base: dict[str, dict[Value, None]] = {}
        indices_by_base: dict[str, set[tuple[object, ...]]] = {}
        for name, values in values_by_variable.items():
            base, texts = self._variable_name_mapper.from_name(name)
            if texts:
                values_by_base.setdefault(base, {}).update(values)
                indices_by_base.setdefault(base, set()).add(tuple(self._index(text) for text in texts))
        offsets_by_arity: dict[int, dict[tuple[int, ...], None]] = {}
        for indices in indices_by_base.values():
            if self._whole(indices):
                offsets = offsets_by_arity.setdefault(self._arity(indices), {})
                for first, second in itertools.permutations(sorted(indices), 2):  # type: ignore[type-var]
                    offsets[tuple(a - b for a, b in zip(first, second, strict=True))] = None  # type: ignore[operator]
        return Vocabulary(
            domain.players.names,
            domain.players.to_act,
            {name: tuple(values) for name, values in values_by_variable.items()},
            {base: tuple(values) for base, values in values_by_base.items()},
            {base: frozenset(indices) for base, indices in indices_by_base.items()},
            {arity: tuple(sorted(offsets)) for arity, offsets in offsets_by_arity.items()},
        )

    def leaves(self, vocabulary: Vocabulary) -> tuple[Expression, ...]:
        expressions: dict[str, Expression] = {}
        for name, values in vocabulary.values_by_variable.items():
            numeric = self._numeric(values)
            for reading in self._readings(name, vocabulary):
                if numeric:
                    self._add(expressions, Expression(reading, 1, 0))
                    continue
                for value in values:
                    renderings = self._rendered(value, vocabulary)
                    if name == vocabulary.to_act:
                        renderings = (repr(value), *renderings)
                    for rendered in renderings:
                        self._add(expressions, Expression(f"{reading} == {rendered}", 1, 0))
        for base, values in vocabulary.values_by_base.items():
            if self._numeric(values):
                for kind in AGGREGATES:
                    if kind != COUNT:
                        aggregate = Aggregate(base, False, kind, f"{VIEW}.{base}[{AGGREGATE_INDEX}]", 1)
                        self._add(expressions, Expression(self._aggregate_template(aggregate), 2, 0, aggregate=aggregate))
                continue
            zero = (0,) * self._arity(vocabulary.indices_by_base[base])
            for value in values:
                for rendered in self._rendered(value, vocabulary):
                    pattern = Pattern(base, (PatternCondition(base, zero, "==", rendered),))
                    self._add(expressions, Expression(self._pattern_template(pattern, vocabulary), 1, 0, pattern))
        for player in (ME, OTHER):
            self._add(expressions, Expression(f"{VIEW}.mobility({player})", 1, 0))
        return tuple(expressions.values())

    def pattern_expression(self, pattern: Pattern, vocabulary: Vocabulary) -> Expression:
        """The expression counting the indices where every condition of the pattern holds; one clause per condition. Every
        base the pattern reads must be in the vocabulary."""
        return Expression(self._pattern_template(pattern, vocabulary), len(pattern.conditions), 0, pattern)

    def pattern_children(self, expression: Expression, vocabulary: Vocabulary) -> tuple[Expression, ...]:
        pattern = expression.pattern
        if pattern is None:
            return ()
        anchor_indices = vocabulary.indices_by_base[pattern.anchor]
        arity, whole = self._arity(anchor_indices), self._whole(anchor_indices)
        zero = (0,) * arity
        offsets = (zero, *vocabulary.offsets_by_arity.get(arity, ())) if whole else (zero,)
        children: dict[str, Expression] = {}
        for base, indices in vocabulary.indices_by_base.items():
            same = indices == anchor_indices
            if self._arity(indices) != arity or not (same or (whole and self._whole(indices))):
                continue
            base_values = vocabulary.values_by_base[base]
            values = (
                []
                if self._numeric(base_values)
                else [rendered for value in base_values for rendered in self._rendered(value, vocabulary)]
            )
            for steps in offsets:
                shifted = [*values, repr(OUTSIDE)] if any(steps) or not same else values
                conditions = [
                    PatternCondition(base, steps, relation, rendered)
                    for relation in PATTERN_RELATIONS
                    for rendered in shifted
                ]
                conditions.extend(
                    PatternCondition(base, steps, relation, None, position)
                    for relation in PATTERN_RELATIONS
                    for position, condition in enumerate(pattern.conditions)
                    if (condition.base, condition.steps) != (base, steps)
                )
                for condition in conditions:
                    if condition in pattern.conditions:
                        continue
                    grown = Pattern(pattern.anchor, (*pattern.conditions, condition))
                    self._add(
                        children,
                        Expression(self._pattern_template(grown, vocabulary), len(grown.conditions), 0, grown),
                    )
        return tuple(children.values())

    def aggregate_children(self, expression: Expression, vocabulary: Vocabulary) -> tuple[Expression, ...]:
        aggregate = expression.aggregate
        if aggregate is None:
            return ()
        indices = vocabulary.indices_by_base[aggregate.base]
        group = [base for base, others in vocabulary.indices_by_base.items() if others == indices]
        body, clauses = aggregate.body, aggregate.body_clauses
        bodies: list[tuple[str, int, bool]] = []
        for pair in (aggregate.pair, True):
            variables = (AGGREGATE_INDEX, AGGREGATE_OTHER_INDEX) if pair else (AGGREGATE_INDEX,)
            if pair and not aggregate.pair:
                other = body.replace(f"[{AGGREGATE_INDEX}]", f"[{AGGREGATE_OTHER_INDEX}]")
                bodies.extend(
                    (template, clauses * 2 + 1, True)
                    for template in (
                        f"({body}) - ({other})",
                        f"abs(({body}) - ({other}))",
                        f"({body}) >= ({other})",
                        f"({body}) == ({other})",
                    )
                )
            readings: list[str] = []
            for base in group:
                values = vocabulary.values_by_base[base]
                for variable in variables:
                    reading = f"{VIEW}.{base}[{variable}]"
                    if self._numeric(values):
                        readings.append(reading)
                    else:
                        readings.extend(
                            f"{reading} == {rendered}" for value in values for rendered in self._rendered(value, vocabulary)
                        )
                if pair and not self._numeric(values):
                    readings.append(f"{VIEW}.{base}[{AGGREGATE_INDEX}] == {VIEW}.{base}[{AGGREGATE_OTHER_INDEX}]")
            if pair and not aggregate.pair:
                continue
            for reading in readings:
                for operation in BODY_OPERATIONS:
                    bodies.append((self._body(operation, body, reading), clauses + 1, pair))
        children: dict[str, Expression] = {}
        for template, body_clauses, pair in bodies:
            for kind in AGGREGATES:
                grown = Aggregate(aggregate.base, pair, kind, template, body_clauses)
                self._add(children, Expression(self._aggregate_template(grown), body_clauses + 1, 0, aggregate=grown))
        return tuple(children.values())

    def unary(self, expression: Expression) -> tuple[tuple[Expression, str], ...]:
        return tuple(
            (Expression(f"{operation}({expression.template})", expression.clauses + 1, expression.plies), operation)
            for operation in UNARY
        )

    def thresholds(self, expression: Expression, values: Sequence[float]) -> tuple[tuple[Expression, str, float], ...]:
        """For each value the expression takes, sorted: at least it, above the lowest, and at most it, below the
        highest."""
        distinct = sorted(set(values))
        children: list[tuple[Expression, str, float]] = []
        for relation in THRESHOLDS:
            cuts = distinct[1:] if relation == ">=" else distinct[:-1]
            for cut in cuts:
                template = f"({expression.template}) {relation} {self._number(cut)}"
                children.append((Expression(template, expression.clauses + 1, expression.plies), relation, cut))
        return tuple(children)

    def combinations(self, expression: Expression, partner: Expression) -> tuple[tuple[Expression, str], ...]:
        first, second = expression.template, partner.template
        clauses, plies = expression.clauses + partner.clauses, max(expression.plies, partner.plies)
        templates = {
            "+": f"({first}) + ({second})",
            "-": f"({first}) - ({second})",
            "*": f"({first}) * ({second})",
            "/": f"({first}) / max(1, {second})",
            "max": f"max({first}, {second})",
            "min": f"min({first}, {second})",
            ">=": f"(({first}) >= ({second}))",
            "==": f"(({first}) == ({second}))",
        }
        return tuple((Expression(templates[operator], clauses, plies), operator) for operator in COMBINATIONS)

    def look_aheads(self, expression: Expression) -> tuple[Expression, ...]:
        variable = f"{LOOK_AHEAD_VARIABLE}{expression.plies + 1}"
        after, here = expression.template.replace(VIEW, variable), expression.template
        clauses, plies = expression.clauses + 1, expression.plies + 1
        children: list[Expression] = []
        for player in (ME, OTHER):
            for body in (after, f"({after}) - ({here})"):
                for kind in (BEST, WORST):
                    children.append(Expression(f"{VIEW}.{kind}({player}, lambda {variable}: {body})", clauses, plies))
            children.append(Expression(f"{VIEW}.{COUNT}({player}, lambda {variable}: {after})", clauses, plies))
            for relation in (">", "<"):
                body = f"({after}) {relation} ({here})"
                children.append(Expression(f"{VIEW}.{COUNT}({player}, lambda {variable}: {body})", clauses, plies))
        return tuple(children)

    def source(self, expression: Expression) -> PythonRule:
        """The expression as a rule reading `here`."""
        return PythonRule(expression.template.replace(VIEW, HERE))

    def _aggregate_template(self, aggregate: Aggregate) -> str:
        loops = f"for {AGGREGATE_INDEX} in {VIEW}.{aggregate.base}"
        if aggregate.pair:
            loops += (
                f" for {AGGREGATE_OTHER_INDEX} in {VIEW}.{aggregate.base} if {AGGREGATE_INDEX} != {AGGREGATE_OTHER_INDEX}"
            )
        if aggregate.kind == COUNT:
            return f"sum(1 {loops} if {aggregate.body})"
        if aggregate.kind == SUM:
            return f"sum(({aggregate.body}) {loops})"
        return f"{aggregate.kind}((({aggregate.body}) {loops}), default=0)"

    def _body(self, operation: str, body: str, reading: str) -> str:
        return {
            "/": f"({body}) / max(1, {reading})",
            "abs": f"abs(({body}) - ({reading}))",
            "and": f"(({body}) and ({reading}))",
            "or": f"(({body}) or ({reading}))",
        }.get(operation, f"({body}) {operation} ({reading})")

    def _numeric(self, values: Sequence[Value]) -> bool:
        return bool(values) and all(isinstance(value, int | float) and not isinstance(value, bool) for value in values)

    def _pattern_template(self, pattern: Pattern, vocabulary: Vocabulary) -> str:
        readings = [self._pattern_reading(pattern.anchor, condition, vocabulary) for condition in pattern.conditions]
        tests = [
            f"{reading} {condition.relation} "
            f"{readings[condition.other_condition] if condition.other_condition is not None else condition.value}"
            for reading, condition in zip(readings, pattern.conditions, strict=True)
        ]
        return f"sum(1 for {PATTERN_INDEX} in {VIEW}.{pattern.anchor} if {' and '.join(tests)})"

    def _pattern_reading(self, anchor: str, condition: PatternCondition, vocabulary: Vocabulary) -> str:
        if not any(condition.steps) and vocabulary.indices_by_base[condition.base] == vocabulary.indices_by_base[anchor]:
            return f"{VIEW}.{condition.base}[{PATTERN_INDEX}]"
        return f"{VIEW}.offset({condition.base!r}, {PATTERN_INDEX}, {', '.join(map(str, condition.steps))})"

    def _readings(self, name: str, vocabulary: Vocabulary) -> tuple[str, ...]:
        """How an expression reads a variable: absolutely, and with each index that is a player's name as `me` or
        `other`."""
        base, texts = self._variable_name_mapper.from_name(name)
        if not texts:
            return (f"{VIEW}.{base}",)
        options = [
            (repr(self._index(text)), *((ME, OTHER) if text in vocabulary.players else ())) for text in texts
        ]
        return tuple(f"{VIEW}.{base}[{', '.join(choice)}]" for choice in itertools.product(*options))

    def _rendered(self, value: Value, vocabulary: Vocabulary) -> tuple[str, ...]:
        return (ME, OTHER) if value in vocabulary.players else (repr(value),)

    def _add(self, expressions: dict[str, Expression], expression: Expression) -> None:
        expressions.setdefault(expression.template, expression)

    def _index(self, text: str) -> object:
        return int(text) if text.lstrip("-").isdecimal() else text

    def _arity(self, indices: frozenset[tuple[object, ...]] | set[tuple[object, ...]]) -> int:
        return len(next(iter(indices)))

    def _whole(self, indices: frozenset[tuple[object, ...]] | set[tuple[object, ...]]) -> bool:
        return all(isinstance(index, int) for indexed in indices for index in indexed)

    def _number(self, value: float) -> str:
        return str(int(value)) if float(value).is_integer() else repr(float(value))
