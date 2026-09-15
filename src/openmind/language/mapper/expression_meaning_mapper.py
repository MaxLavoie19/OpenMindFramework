import ast

from openmind.inference.constant.inference_constant import (
    AGGREGATE_INDEX,
    AGGREGATE_OTHER_INDEX,
    BEST,
    COUNT,
    HERE,
    ME,
    OTHER,
    OUTSIDE,
    PATTERN_INDEX,
    WORST,
)
from openmind.language.constant import meaning_constant as meaning
from openmind.language.model.glossary import Glossary
from openmind.language.model.meaning import Meaning

#: Each view's name with the players who moved to reach its position, and the innermost view's name.
type Views = tuple[dict[str, list[str]], str]

_PLAYERS = {ME: meaning.ME, OTHER: meaning.OPPONENT}
_LOOK_AHEADS = {BEST: meaning.HIGHEST, WORST: meaning.LOWEST, COUNT: meaning.COUNT}
_AGGREGATES = {"sum": meaning.TOTAL, "min": meaning.LOWEST, "max": meaning.HIGHEST}


class ExpressionMeaningMapper:
    """Maps an expression's Python source to its meaning, construct by construct, for any domain: what it reads, where,
    on which position, and how it combines them. A reading on the position the innermost look-ahead reaches has no
    position; one on an earlier position has the players who moved to reach it (none: now). A subtraction of the same
    reading on two positions is a change. An index the glossary labels becomes its label. A construct without a meaning
    is kept as its source, so nothing is dropped."""

    def to_meaning(self, source: str, glossary: Glossary | None = None) -> Meaning:
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError:
            return {meaning.SOURCE: source.strip()}
        return self._meaning(tree.body, ({HERE: []}, HERE), {}, glossary or Glossary())

    def _meaning(self, node: ast.expr, views: Views, places: dict[str, Meaning], glossary: Glossary) -> Meaning:
        match node:
            case ast.Constant(value=value):
                return self._value(value)
            case ast.Name(id=name) if name in places:
                return places[name]
            case ast.Name(id=name) if name in _PLAYERS:
                return {meaning.PLAYER: _PLAYERS[name]}
            case ast.Compare(left=left, ops=[operator], comparators=[right]) if type(operator).__name__ in meaning.COMPARISONS:
                return {
                    meaning.COMPARE: meaning.COMPARISONS[type(operator).__name__],
                    meaning.LEFT: self._meaning(left, views, places, glossary),
                    meaning.RIGHT: self._meaning(right, views, places, glossary),
                }
            case ast.BoolOp(op=operator, values=values):
                key = meaning.ALL_OF if isinstance(operator, ast.And) else meaning.ANY_OF
                return {key: [self._meaning(value, views, places, glossary) for value in values]}
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                return {meaning.NOT: self._meaning(operand, views, places, glossary)}
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return {meaning.NEGATIVE: self._meaning(operand, views, places, glossary)}
            case ast.BinOp(op=ast.Sub(), left=left, right=right) if self._normalized(left, views) == self._normalized(right, views):
                return {meaning.CHANGE_IN: self._meaning(left, views, places, glossary)}
            case ast.BinOp(op=operator, left=left, right=right) if type(operator).__name__ in meaning.OPERATIONS:
                return {
                    meaning.ARITHMETIC: meaning.OPERATIONS[type(operator).__name__],
                    meaning.LEFT: self._meaning(left, views, places, glossary),
                    meaning.RIGHT: self._meaning(right, views, places, glossary),
                }
            case ast.Subscript(value=ast.Attribute(value=ast.Name(id=view), attr=base), slice=index) if view in views[0]:
                return self._positioned(
                    {meaning.READING: base, meaning.AT: self._index(base, index, places, glossary)}, view, views
                )
            case ast.Attribute(value=ast.Name(id=view), attr=base) if view in views[0]:
                return self._positioned({meaning.READING: base}, view, views)
            case ast.Call():
                called = self._call(node, views, places, glossary)
                if called is not None:
                    return called
        return {meaning.SOURCE: ast.unparse(node)}

    def _call(self, node: ast.Call, views: Views, places: dict[str, Meaning], glossary: Glossary) -> Meaning | None:
        match node:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=view), attr=kind),
                args=[ast.Name(id=player), ast.Lambda(args=ast.arguments(args=[ast.arg(arg=after)]), body=body)],
            ) if view in views[0] and kind in _LOOK_AHEADS and player in _PLAYERS:
                inner: Views = ({**views[0], after: [*views[0][view], _PLAYERS[player]]}, after)
                return {
                    meaning.LOOK_AHEAD: _LOOK_AHEADS[kind],
                    meaning.MOVES_OF: _PLAYERS[player],
                    meaning.OF: self._meaning(body, inner, places, glossary),
                }
            case ast.Call(func=ast.Attribute(value=ast.Name(id=view), attr="mobility"), args=[ast.Name(id=player)]) if (
                view in views[0] and player in _PLAYERS
            ):
                return self._positioned({meaning.MOVES_AVAILABLE_TO: _PLAYERS[player]}, view, views)
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=view), attr="offset"),
                args=[ast.Constant(value=str() as base), ast.Name(id=place), *steps],
            ) if view in views[0] and place in places and all(isinstance(step, ast.Constant | ast.UnaryOp) for step in steps):
                return self._positioned(
                    {
                        meaning.READING: base,
                        meaning.STEPS: [ast.literal_eval(step) for step in steps],
                        meaning.FROM: places[place],
                    },
                    view,
                    views,
                )
            case ast.Call(func=ast.Name(id="sum" | "min" | "max" as kind), args=[ast.GeneratorExp() as generator]):
                return self._aggregate(kind, generator, views, places, glossary)
            case ast.Call(func=ast.Name(id="max"), args=[ast.Constant(value=1), second]):
                return {meaning.AT_LEAST_ONE: self._meaning(second, views, places, glossary)}
            case ast.Call(func=ast.Name(id="max" | "min" as kind), args=[first, second]):
                key = meaning.LARGER_OF if kind == "max" else meaning.SMALLER_OF
                return {key: [self._meaning(first, views, places, glossary), self._meaning(second, views, places, glossary)]}
            case ast.Call(func=ast.Name(id="abs"), args=[ast.BinOp(op=ast.Sub(), left=first, right=second)]):
                return {
                    meaning.DISTANCE_BETWEEN: [
                        self._meaning(first, views, places, glossary),
                        self._meaning(second, views, places, glossary),
                    ]
                }
            case ast.Call(func=ast.Name(id="abs"), args=[body]):
                return {meaning.SIZE_OF: self._meaning(body, views, places, glossary)}
            case ast.Call(func=ast.Name(id="wins"), args=[ast.Name(id=player), *_]) if player in _PLAYERS:
                return {meaning.WINNING_MOVES_OF: _PLAYERS[player]}
            case ast.Call(func=ast.Name(id="solo_distance"), args=[ast.Name(id=player), *_]) if player in _PLAYERS:
                return {meaning.MOVES_TO_A_WIN_FOR: _PLAYERS[player]}
            case ast.Call(func=ast.Name(id="win_chance")):
                return {meaning.WIN_CHANCE_OF_THE_ACTION: True}
        return None

    def _aggregate(
        self, kind: str, generator: ast.GeneratorExp, views: Views, places: dict[str, Meaning], glossary: Glossary
    ) -> Meaning | None:
        loops = generator.generators
        targets = [loop.target.id for loop in loops if isinstance(loop.target, ast.Name)]
        if len(targets) != len(loops):
            return None
        conditions = [
            condition
            for loop in loops
            for condition in loop.ifs
            if not (isinstance(condition, ast.Compare) and isinstance(condition.ops[0], ast.NotEq) and len(targets) == 2)
        ]
        counting = kind == "sum" and isinstance(generator.elt, ast.Constant) and generator.elt.value == 1
        if targets == [PATTERN_INDEX] and counting:
            inner = {**places, PATTERN_INDEX: {meaning.PLACE: meaning.THERE}}
            return {meaning.COUNT: meaning.PLACES, meaning.WHERE: self._all(conditions, views, inner, glossary)}
        if targets == [AGGREGATE_INDEX]:
            over, inner = meaning.ENTRIES, {**places, AGGREGATE_INDEX: {meaning.ENTRY: meaning.THAT_ENTRY}}
        elif targets == [AGGREGATE_INDEX, AGGREGATE_OTHER_INDEX]:
            over = meaning.PAIRS
            inner = {
                **places,
                AGGREGATE_INDEX: {meaning.ENTRY: meaning.ONE_ENTRY},
                AGGREGATE_OTHER_INDEX: {meaning.ENTRY: meaning.OTHER_ENTRY},
            }
        else:
            return None
        if counting:
            return {
                meaning.AGGREGATE: meaning.COUNT,
                meaning.OVER: over,
                meaning.WHERE: self._all(conditions, views, inner, glossary),
            }
        if conditions:
            return None
        return {
            meaning.AGGREGATE: _AGGREGATES[kind],
            meaning.OVER: over,
            meaning.OF: self._meaning(generator.elt, views, inner, glossary),
        }

    def _all(self, conditions: list[ast.expr], views: Views, places: dict[str, Meaning], glossary: Glossary) -> list[Meaning]:
        return [self._meaning(condition, views, places, glossary) for condition in conditions]

    def _index(self, base: str, index: ast.expr, places: dict[str, Meaning], glossary: Glossary) -> Meaning:
        match index:
            case ast.Name(id=place) if place in places:
                return places[place]
            case ast.Name(id=player) if player in _PLAYERS:
                return {meaning.PLAYER: _PLAYERS[player]}
            case ast.Tuple(elts=elements) if all(isinstance(element, ast.Constant) for element in elements):
                return self._labelled(base, tuple(element.value for element in elements), glossary)  # type: ignore[attr-defined]
            case ast.Constant(value=value):
                return self._labelled(base, (value,), glossary)
        return {meaning.SOURCE: ast.unparse(index)}

    def _labelled(self, base: str, indices: tuple[object, ...], glossary: Glossary) -> Meaning:
        label = glossary.labels.get(base, {}).get(indices)
        return {meaning.LABEL: label} if label is not None else {meaning.INDICES: list(indices)}

    def _positioned(self, reading: Meaning, view: str, views: Views) -> Meaning:
        if view == views[1]:
            return reading
        moved = views[0][view]
        return {**reading, meaning.POSITION: {meaning.NOW: True} if not moved else {meaning.AFTER_MOVES_OF: list(moved)}}

    def _normalized(self, node: ast.expr, views: Views) -> str:
        """The node with every view's name alike, so the same reading on two positions compares equal."""
        copy = ast.parse(ast.unparse(node), mode="eval")
        for child in ast.walk(copy):
            if isinstance(child, ast.Name) and child.id in views[0]:
                child.id = HERE
        return ast.dump(copy)

    def _value(self, value: object) -> Meaning:
        if value is None:
            return {meaning.EMPTY: True}
        if value == OUTSIDE:
            return {meaning.OUTSIDE: True}
        if isinstance(value, bool):
            return {meaning.TRUTH: value}
        if isinstance(value, int | float):
            return {meaning.NUMBER: int(value) if float(value).is_integer() else value}
        return {meaning.NAME: str(value)}
