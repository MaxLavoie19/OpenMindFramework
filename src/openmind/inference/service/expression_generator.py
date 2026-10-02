import itertools
from collections.abc import Iterable, Sequence

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
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.world.model.state import State
from openmind.structure.model.value import Value


class ExpressionGenerator:
    """Generates expressions for any domain from the positions of its rows and its players only; nothing here knows a
    game. A player's name, as a value or as an index, is written relative to the player valued: `me` or `other`.

    Values are either names (a mark, a piece, a player, None) or numbers (a position, a clock, a count); a variable or
    base is numeric when every value seen is a number, and a number is never compared for equality with a value seen.

    - Leaves: a numeric variable as it is (`{view}.x[me]`); every other variable at every value seen
      (`{view}.cell[2, 2] == me`), the player to act also absolutely (`{view}.turn == 'X'`); for every indexed base of
      names and value, how many indices hold it, a pattern of one condition; for every numeric indexed base, the sum,
      lowest and highest of its values, aggregates of one reading; and how many actions each player could take.
    - A pattern grows by one condition: any base with the anchor's indices read at the pattern's index, or, over a
      grid, any grid of as many dimensions read at the pattern's index shifted by any offset seen between two cells
      (the same cell included); equal or not equal to any name its base takes, to OUTSIDE, or to the variable another
      condition reads.
    - An aggregate's body grows by an operation (`+ - * / abs >= <= == and or`) with a reading of any base sharing its
      indices, at `i`, or at `j` once it reads pairs; a body at `i` also turns into a comparison of itself at `i` and at
      `j`, over every pair of different indices. Each body is summed, counted, and taken at its lowest and highest.
    - Thresholds: an expression at least, or at most, each value it takes; and its absolute value.
    - Combinations of two expressions: `+`, `-`, `*`, `/ max(1, ·)`, `max`, `min`, `>=`, `==`.
    - Look-aheads, for `me` and for `other`: the best, the worst and the count of the expression after an action, and
      the best and worst change of the expression, and how many actions raise or lower it."""

    def vocabulary(self, rbs: RuleBasedGame, states: Iterable[State]) -> Vocabulary:
        """Every scalar, grid cell and map entry of the states with the values seen; lists aren't read."""
        values_by_variable: dict[tuple[str, tuple[Value, ...]], dict[Value, None]] = {}
        values_by_base: dict[str, dict[Value, None]] = {}
        indices_by_base: dict[str, set[tuple[object, ...]]] = {}
        grids: set[str] = set()
        for state in states:
            for name, model in state.models:
                if isinstance(model, Scalar):
                    values_by_variable.setdefault((name, ()), {})[model.value] = None
                    continue
                if isinstance(model, Grid):
                    grids.add(name)
                    entries: Iterable[tuple[tuple[Value, ...], Value]] = model.items()
                elif isinstance(model, Map):
                    entries = (((key,), value) for key, value in model.items)
                else:
                    continue
                for index, value in entries:
                    values_by_variable.setdefault((name, index), {})[value] = None
                    values_by_base.setdefault(name, {})[value] = None
                    indices_by_base.setdefault(name, set()).add(index)
        offsets_by_arity: dict[int, dict[tuple[int, ...], None]] = {}
        for base in sorted(grids):
            indices = indices_by_base[base]
            offsets = offsets_by_arity.setdefault(self._arity(indices), {})
            for first, second in itertools.permutations(sorted(indices), 2):  # type: ignore[type-var]
                offsets[tuple(a - b for a, b in zip(first, second, strict=True))] = None  # type: ignore[operator]
        return Vocabulary(
            rbs.players().names,
            {variable: tuple(values) for variable, values in values_by_variable.items()},
            {base: tuple(values) for base, values in values_by_base.items()},
            {base: frozenset(indices) for base, indices in indices_by_base.items()},
            {arity: tuple(sorted(offsets)) for arity, offsets in offsets_by_arity.items()},
            frozenset(grids),
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
                    for rendered in renderings:
                        self._add(expressions, Expression(f"{reading} == {rendered}", 1, 0))
        for base, values in vocabulary.values_by_base.items():
            if self._numeric(values):
                for kind in AGGREGATES:
                    if kind != COUNT:
                        reading = f"{VIEW}.{base}[{AGGREGATE_INDEX}]"
                        aggregate = Aggregate(base, False, kind, reading, 1, readings=(reading,))
                        self._add(expressions, Expression(self._aggregate_template(aggregate), 2, 0, aggregate=aggregate))
                continue
            indices = vocabulary.indices_by_base[base]
            zero = (0,) * self._arity(indices)
            for value in values:
                for rendered in self._rendered(value, vocabulary):
                    pattern = Pattern(base, (PatternCondition(base, zero, "==", rendered),))
                    self._add(expressions, Expression(self._pattern_template(pattern, vocabulary), 1, 0, pattern))
                    if base in vocabulary.grids:
                        held = f"{VIEW}.{base}[{AGGREGATE_INDEX}] == {rendered}"
                        readings = tuple((self._changed(base, player, AGGREGATE_INDEX), 1) for player in (ME, OTHER))
                        for reading, plies in readings:
                            aggregate = Aggregate(
                                base,
                                False,
                                COUNT,
                                self._body("and", held, reading),
                                2,
                                plies,
                                readings=(held, reading),
                                operations=("and",),
                            )
                            self._add(expressions, Expression(self._aggregate_template(aggregate), 3, plies, aggregate=aggregate))
        for player in (ME, OTHER):
            self._add(expressions, Expression(f"{VIEW}.mobility({player})", 1, 0))
        return tuple(expressions.values())

    def pattern_expression(self, pattern: Pattern, vocabulary: Vocabulary) -> Expression:
        """The expression counting the indices where every condition holds and no absence does; one clause per condition
        and one per condition denied. Every base the pattern reads must be in the vocabulary.

        **The denials are counted, because a price that cannot see them buys complexity for free.** What a
        term costs is what the fit charges to keep it, and a pattern denying ten pairs is twenty readings
        whoever built it."""
        return Expression(self._pattern_template(pattern, vocabulary), self._clauses(pattern), 0, pattern)

    def counting(
        self, base: str, value: Value, vocabulary: Vocabulary, owner: str | None = None
    ) -> tuple[Expression, ...]:
        """The expressions counting how many of that value stand in that base — and, where a base says whose each
        one is, how many of them are the reader's own.

        **The second is the one that is worth anything, and the search reaches it late.** Where a game keeps what
        a thing is and whose it is in two structures over the same places — chess holds `piece` and `color` that
        way — counting a value alone counts both sides' knights at once. That number barely moves from position
        to position, so it explains nothing and the search passes over it; the term whose weight *is* what a
        knight is worth needs the second condition, and the search only reaches it by growing a child of the
        near-constant parent it already declined. Both are offered here so the seeding does not depend on that
        happening.

        Nothing about it is a board or a piece. It is how many of a thing there are, and how many of them are
        mine, for a game that has things and says whose they are — and a game that says neither produces
        neither."""
        if base not in vocabulary.indices_by_base or value not in vocabulary.values_by_base.get(base, ()):
            return ()
        zero = (0,) * self._arity(vocabulary.indices_by_base[base])
        found = [
            self.pattern_expression(Pattern(base, (PatternCondition(base, zero, "==", rendered),)), vocabulary)
            for rendered in self._rendered(value, vocabulary)
        ]
        if owner is None or owner not in vocabulary.indices_by_base:
            return tuple(found)
        if vocabulary.indices_by_base[owner] != vocabulary.indices_by_base[base]:
            return tuple(found)
        found.extend(
            self.pattern_expression(
                Pattern(
                    base,
                    (PatternCondition(base, zero, "==", rendered), PatternCondition(owner, zero, "==", ME)),
                ),
                vocabulary,
            )
            for rendered in self._rendered(value, vocabulary)
        )
        return tuple(found)

    def owning(self, base: str, vocabulary: Vocabulary) -> str | None:
        """Which base says whose the things in that one are: another over the same places, holding players' names.

        Worked out and never declared, so a game that keeps ownership somewhere else, or nowhere, is not being
        told it does. Where several would do, the first by name is taken and the rest are as good — each gives a
        term, and what a term is worth is the fitter's to find."""
        if base not in vocabulary.indices_by_base or not vocabulary.players:
            return None
        places = vocabulary.indices_by_base[base]
        return next(
            (
                name
                for name, indices in sorted(vocabulary.indices_by_base.items())
                if name != base
                and indices == places
                and any(one in vocabulary.players for one in vocabulary.values_by_base.get(name, ()))
            ),
            None,
        )

    def pattern_children(self, expression: Expression, vocabulary: Vocabulary) -> tuple[Expression, ...]:
        pattern = expression.pattern
        if pattern is None:
            return ()
        anchor_indices = vocabulary.indices_by_base[pattern.anchor]
        arity, whole = self._arity(anchor_indices), pattern.anchor in vocabulary.grids
        zero = (0,) * arity
        offsets = (zero, *vocabulary.offsets_by_arity.get(arity, ())) if whole else (zero,)
        children: dict[str, Expression] = {}
        # Every candidate condition, kept across the bases rather than used inside one. An absence pairs two
        # bases at one offset — what a thing is and whose it is — so gathering them per base and pairing
        # inside that loop would never put two bases together, which is the whole of what an absence says.
        offered: list[PatternCondition] = []
        for base, indices in vocabulary.indices_by_base.items():
            same = indices == anchor_indices
            if self._arity(indices) != arity or not (same or (whole and base in vocabulary.grids)):
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
                    grown = Pattern(pattern.anchor, (*pattern.conditions, condition), pattern.absences)
                    self._add(children, self.pattern_expression(grown, vocabulary))
        return tuple(children.values())

    #: **Absences are not grown here, and that is measured rather than assumed.**
    #:
    #: `Pattern.absences` exists because nine of the Chess Intelligence Agent's terms need one, and a pattern
    #: can hold them — but growing them a pair at a time was 40% of the children of every pattern, 85 against
    #: 51, in a search its clock already holds to two generations. What that bought, over a store of 1,189
    #: fitted position rules: not one of them held a denial.
    #:
    #: The reason is the clearest finding of the literature on this: no system discovers which negated
    #: conjunction to use, and Progol and Aleph both require one supplied as a background predicate. No
    #: shorter version of an absence predicts anything, so there is nothing for a search to climb toward —
    #: the same reason a passed pawn is unreachable by growth.
    #:
    #: So absences are assembled rather than searched. `TermAssembler` chooses them against what a heuristic
    #: is missing, where each is kept because it accounts for something, and a passed pawn comes out of ninety
    #: boards at 0.927 on boards it never read. The search keeps its budget for terms it can reach.

    def _clauses(self, pattern: Pattern) -> int:
        """What that pattern costs: one for each condition that must hold, one for each condition denied."""
        return len(pattern.conditions) + sum(len(group) for group in pattern.absences)

    def aggregate_children(self, expression: Expression, vocabulary: Vocabulary) -> tuple[Expression, ...]:
        aggregate = expression.aggregate
        if aggregate is None:
            return ()
        indices = vocabulary.indices_by_base[aggregate.base]
        group = [base for base, others in vocabulary.indices_by_base.items() if others == indices]
        whole = aggregate.base in vocabulary.grids
        body, clauses, body_plies = aggregate.body, aggregate.body_clauses, aggregate.body_plies
        bodies: list[tuple[str, int, bool, int, tuple[str, ...], tuple[str, ...]]] = []
        for pair in (aggregate.pair, True):
            variables = (AGGREGATE_INDEX, AGGREGATE_OTHER_INDEX) if pair else (AGGREGATE_INDEX,)
            if pair and not aggregate.pair:
                other = self._at_other_index(body)
                bodies.extend(
                    (template, clauses * 2 + 1, True, body_plies, (), ())
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
                        readings.extend(self._what_if_readings(base, variable, renderings))
                        readings.extend(self._exchange_readings(base, variable, renderings))
                if pair and not self._numeric(values):
                    readings.append((f"{VIEW}.{base}[{AGGREGATE_INDEX}] == {VIEW}.{base}[{AGGREGATE_OTHER_INDEX}]", 0))
                if pair and whole:
                    if ME in renderings:
                        copied = f"{VIEW}.copied({AGGREGATE_OTHER_INDEX}, {AGGREGATE_INDEX})"
                        readings.extend(
                            (f"{copied}.changed({player}, {base!r}, {AGGREGATE_INDEX})", 1) for player in (ME, OTHER)
                        )
            if pair and not aggregate.pair:
                continue
            for reading, plies in readings:
                for operation in BODY_OPERATIONS:
                    grown = (*aggregate.readings, reading) if aggregate.readings else ()
                    bodies.append(
                        (
                            self._body(operation, body, reading),
                            clauses + 1,
                            pair,
                            max(body_plies, plies),
                            grown,
                            (*aggregate.operations, operation) if grown else (),
                        )
                    )
        children: dict[str, Expression] = {}
        for template, grown_clauses, pair, grown_plies, parts, operations in bodies:
            for kind in AGGREGATES:
                grown = Aggregate(
                    aggregate.base, pair, kind, template, grown_clauses, grown_plies, readings=parts, operations=operations
                )
                self._add(children, Expression(self._aggregate_template(grown), grown_clauses + 1, grown_plies, aggregate=grown))
        return tuple(children.values())

    def _changed(self, base: str, player: str, index: str) -> str:
        """How many of the player's actions change the base at the index: one action ahead."""
        return f"{VIEW}.changed({player}, {base!r}, {index})"

    def _what_if_readings(self, base: str, index: str, renderings: Sequence[str]) -> list[tuple[str, int]]:
        """What if: for a grid whose values name players, how many of a player's moves would change the cell if it held
        the other player's name."""
        readings: list[tuple[str, int]] = []
        if ME in renderings:
            readings.extend(
                (f"{VIEW}.with_value({base!r}, {index}, {owner}).changed({player}, {base!r}, {index})", 1)
                for owner, player in ((OTHER, ME), (ME, OTHER))
            )
        return readings

    def _exchange_readings(self, base: str, index: str, renderings: Sequence[str]) -> list[tuple[str, int]]:
        """Take on this square, and then what can still be done to it: an exchange, without knowing what one is.

        **The narrow look-ahead, which is the one a tactic needs.** `look_aheads` reads the position after
        every move a player has; this reads it after the moves that touch one square, and asks what the other
        side can then do to that same square. Of the moves that take here, the one after which they can take
        back least is the question "can I take this and keep it" — and nobody had to say what a capture is, or
        that a square can be defended, for that to be askable.

        Two plies: the move, and counting what answers it. Nesting one of these inside another's reading is
        the rest of the chain, which the search grows the way it grows everything else.

        Only where the base says whose a thing is, since an exchange is about ownership changing hands."""
        if ME not in renderings:
            return []
        variable = f"{LOOK_AHEAD_VARIABLE}{index}"
        return [
            (
                f"{VIEW}.{kind}_changing({mover}, {base!r}, {index}, "
                f"lambda {variable}: {variable}.changed({answering}, {base!r}, {index}))",
                2,
            )
            for kind in (BEST, WORST)
            for mover, answering in ((ME, OTHER), (OTHER, ME))
        ]

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
            reading if condition.says is not None else
            f"{reading} {condition.relation} "
            f"{readings[condition.other_condition] if condition.other_condition is not None else condition.value}"
            for reading, condition in zip(readings, pattern.conditions, strict=True)
        ]
        # **And what must not be there**, which is the one thing the conjunction above cannot say. Each group
        # is a conjunction in its own right and the whole of it is denied, so a group of two says "not a pawn
        # of theirs" where two plain conditions would say "not a pawn, and not theirs" — a different and much
        # stronger claim that also rules out a pawn of mine.
        tests.extend(
            f"not ({' and '.join(self._absent(pattern, group, readings, vocabulary))})"
            for group in pattern.absences
        )
        return f"sum(1 for {PATTERN_INDEX} in {VIEW}.{pattern.anchor} if {' and '.join(tests)})"

    def _absent(
        self, pattern: Pattern, group: Sequence[PatternCondition], readings: Sequence[str], vocabulary: Vocabulary
    ) -> list[str]:
        """One absence group's conditions, read the same way the kept conditions are.

        A condition of a group may still point at a kept condition's reading, which is why `readings` comes in
        rather than being worked out again here."""
        return [
            self._pattern_reading(pattern.anchor, condition, vocabulary) if condition.says is not None else
            f"{self._pattern_reading(pattern.anchor, condition, vocabulary)} {condition.relation} "
            f"{readings[condition.other_condition] if condition.other_condition is not None else condition.value}"
            for condition in group
        ]

    def _pattern_reading(self, anchor: str, condition: PatternCondition, vocabulary: Vocabulary) -> str:
        if condition.says is not None:
            return condition.says
        if not any(condition.steps) and vocabulary.indices_by_base[condition.base] == vocabulary.indices_by_base[anchor]:
            return f"{VIEW}.{condition.base}[{PATTERN_INDEX}]"
        return f"{VIEW}.offset({condition.base!r}, {PATTERN_INDEX}, {', '.join(map(str, condition.steps))})"

    def _readings(self, variable: tuple[str, tuple[Value, ...]], vocabulary: Vocabulary) -> tuple[str, ...]:
        """How an expression reads a variable: absolutely, and with each index that is a player's name as `me` or
        `other`. A cell of a grid of one dimension is read with its coordinates as a tuple: `{view}.cell[3,]`."""
        base, index = variable
        named = self._named(base)
        if not index:
            return (named,)
        options = [(repr(part), *((ME, OTHER) if part in vocabulary.players else ())) for part in index]
        trailing = "," if base in vocabulary.grids and len(index) == 1 else ""
        return tuple(f"{named}[{', '.join(choice)}{trailing}]" for choice in itertools.product(*options))

    def _named(self, base: str) -> str:
        """How an expression names that model: as an attribute where the name is one Python allows, and subscripted
        where it is not.

        A game is free to name what it holds so that a person can read it — "black may castle king side" — and those
        names are offered as readings. An expression is Python source, so a name with a space in it has to be reached
        by subscript or the whole expression fails to compile."""
        return f"{VIEW}.{base}" if base.isidentifier() else f"{VIEW}[{base!r}]"

    def _rendered(self, value: Value, vocabulary: Vocabulary) -> tuple[str, ...]:
        return (ME, OTHER) if value in vocabulary.players else (repr(value),)

    def _add(self, expressions: dict[str, Expression], expression: Expression) -> None:
        expressions.setdefault(expression.template, expression)

    def _arity(self, indices: frozenset[tuple[object, ...]] | set[tuple[object, ...]]) -> int:
        return len(next(iter(indices)))

    def _number(self, value: float) -> str:
        return str(int(value)) if float(value).is_integer() else repr(float(value))
