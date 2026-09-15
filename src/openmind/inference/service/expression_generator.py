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
            indices = vocabulary.indices_by_base[base]
            zero = (0,) * self._arity(indices)
            for value in values:
                for rendered in self._rendered(value, vocabulary):
                    pattern = Pattern(base, (PatternCondition(base, zero, "==", rendered),))
                    self._add(expressions, Expression(self._pattern_template(pattern, vocabulary), 1, 0, pattern))
                    if self._whole(indices):
                        held = f"{VIEW}.{base}[{AGGREGATE_INDEX}] == {rendered}"
                        readings = (
                            (self._parity(base, AGGREGATE_INDEX), 0),
                            *((self._changed(base, player, AGGREGATE_INDEX), 1) for player in (ME, OTHER)),
                        )
                        for reading, plies in readings:
                            aggregate = Aggregate(base, False, COUNT, self._body("and", held, reading), 2, plies)
                            self._add(expressions, Expression(self._aggregate_template(aggregate), 3, plies, aggregate=aggregate))
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
        whole = self._whole(indices)
        body, clauses, body_plies = aggregate.body, aggregate.body_clauses, aggregate.body_plies
        bodies: list[tuple[str, int, bool, int]] = []
        for pair in (aggregate.pair, True):
            variables = (AGGREGATE_INDEX, AGGREGATE_OTHER_INDEX) if pair else (AGGREGATE_INDEX,)
            if pair and not aggregate.pair:
                other = self._at_other_index(body)
                bodies.extend(
                    (template, clauses * 2 + 1, True, body_plies)
                    for template in (
                        f"({body}) - ({other})",
                        f"abs(({body}) - ({other}))",
                        f"({body}) >= ({other})",
                        f"({body}) == ({other})",
                    )
                )
            readings: list[tuple[str, int]] = []
            for base in group:
                values = vocabulary.values_by_base[base]
                renderings = [] if self._numeric(values) else [rendered for value in values for rendered in self._rendered(value, vocabulary)]
                for variable in variables:
                    reading = f"{VIEW}.{base}[{variable}]"
                    if self._numeric(values):
                        readings.append((reading, 0))
                    else:
                        readings.extend((f"{reading} == {rendered}", 0) for rendered in renderings)
                    if whole:
                        readings.extend((self._changed(base, player, variable), 1) for player in (ME, OTHER))
                        readings.extend(self._cell_readings(base, variable, renderings, self._arity(indices)))
                        readings.extend(self._what_if_readings(base, variable, renderings))
                if pair and not self._numeric(values):
                    readings.append((f"{VIEW}.{base}[{AGGREGATE_INDEX}] == {VIEW}.{base}[{AGGREGATE_OTHER_INDEX}]", 0))
                if pair and whole:
                    readings.extend(
                        (f"({rendered} in {VIEW}.{base}.between({AGGREGATE_INDEX}, {AGGREGATE_OTHER_INDEX}))", 0) for rendered in renderings
                    )
                    if ME in renderings:
                        copied = f"{VIEW}.copied({AGGREGATE_OTHER_INDEX}, {AGGREGATE_INDEX})"
                        readings.extend(
                            (f"{copied}.changed({player}, {base!r}, {AGGREGATE_INDEX})", 1) for player in (ME, OTHER)
                        )
            if whole:
                readings.extend((self._parity(aggregate.base, variable), 0) for variable in variables)
                if pair:
                    readings.extend(
                        (f"{VIEW}.{aggregate.base}.{query}({AGGREGATE_INDEX}, {AGGREGATE_OTHER_INDEX})", 0)
                        for query in ("distance", "aligned")
                    )
            if pair and not aggregate.pair:
                continue
            for reading, plies in readings:
                for operation in BODY_OPERATIONS:
                    bodies.append((self._body(operation, body, reading), clauses + 1, pair, max(body_plies, plies)))
        children: dict[str, Expression] = {}
        for template, grown_clauses, pair, grown_plies in bodies:
            for kind in AGGREGATES:
                grown = Aggregate(aggregate.base, pair, kind, template, grown_clauses, grown_plies)
                self._add(children, Expression(self._aggregate_template(grown), grown_clauses + 1, grown_plies, aggregate=grown))
        return tuple(children.values())

    def _parity(self, base: str, index: str) -> str:
        """Which of two alternating colours the cell at the index is."""
        return f"{VIEW}.{base}.parity({index})"

    def _changed(self, base: str, player: str, index: str) -> str:
        """How many of the player's actions change the base at the index: one action ahead."""
        return f"{VIEW}.changed({player}, {base!r}, {index})"

    def _cell_readings(self, base: str, index: str, renderings: Sequence[str], arity: int) -> list[tuple[str, int]]:
        """Around the cell at the index: how many neighbours hold each name, and whether a ray in each direction meets
        it; a direction steps every coordinate by -1, 0 or 1."""
        readings = [
            (f"sum(1 for value in {VIEW}.{base}.neighbours({index}) if value == {rendered})", 0) for rendered in renderings
        ]
        for direction in itertools.product((-1, 0, 1), repeat=arity):
            if any(direction):
                readings.extend((f"({rendered} in {VIEW}.{base}.ray({index}, {direction!r}))", 0) for rendered in renderings)
        return readings

    def _what_if_readings(self, base: str, index: str, renderings: Sequence[str]) -> list[tuple[str, int]]:
        """What if: how many moves the thing at the index has alone on the grids, for each player; and, for a grid whose
        values name players, how many of a player's moves would change the cell if it held the other player's name."""
        readings = [(f"{VIEW}.alone({index}).mobility({player})", 1) for player in (ME, OTHER)]
        if ME in renderings:
            readings.extend(
                (f"{VIEW}.with_value({base!r}, {index}, {owner}).changed({player}, {base!r}, {index})", 1)
                for owner, player in ((OTHER, ME), (ME, OTHER))
            )
        return readings

    def _at_other_index(self, body: str) -> str:
        """The body read at `j` where it reads `i`: as an index, or as a call's only, first, middle or last argument."""
        for form in ("[{0}]", "({0})", "({0},", ", {0},", ", {0})"):
            body = body.replace(form.format(AGGREGATE_INDEX), form.format(AGGREGATE_OTHER_INDEX))
        return body

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
