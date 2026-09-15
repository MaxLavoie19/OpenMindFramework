import ast

from openmind.inference.constant.inference_constant import (
    AGGREGATE_INDEX,
    AGGREGATE_OTHER_INDEX,
    BEST,
    COUNT,
    HERE,
    OUTSIDE,
    PATTERN_INDEX,
    WORST,
)
from openmind.inference.constant.sentence_constant import (
    AFTER_MOVE,
    AGGREGATE_TEMPLATES,
    ARITHMETIC_WORDS,
    AT_LEAST_ONE_TEMPLATE,
    AT_TEMPLATE,
    CHANGE_TEMPLATE,
    COMPARISON_WORDS,
    CONNECTIVE_WORDS,
    DISTANCE_TEMPLATE,
    EMPTY_WORD,
    ENTRIES,
    EVERY_ENTRY,
    EVERY_PAIR,
    LARGER_TEMPLATE,
    LOOK_AHEAD_TEMPLATES,
    MOBILITY_TEMPLATE,
    NEGATIVE_TEMPLATE,
    NOT_TEMPLATE,
    NOW_WORD,
    OF_TEMPLATE,
    OFFSET_TEMPLATE,
    ONE_ENTRY,
    OTHER_ENTRY,
    OUTSIDE_WORD,
    OWNED_TEMPLATE,
    PAIRS,
    PATTERN_TEMPLATE,
    PLACE,
    PLAYER_VALUES,
    PLAYER_WORDS,
    QUALIFIED_TEMPLATE,
    READING_TEMPLATE,
    SIZE_TEMPLATE,
    SMALLER_TEMPLATE,
    SOLO_DISTANCE_TEMPLATE,
    SOURCE_TEMPLATE,
    THAT_ENTRY,
    THEN_MOVE,
    WHETHER_TEMPLATE,
    WIN_CHANCE_TEMPLATE,
    WINS_TEMPLATE,
)

#: Where a reading is read: each view's name with the words for its position, and the innermost view's name.
type Views = tuple[dict[str, str], str]


class ExpressionSentenceMapper:
    """Reads an expression's Python source as a literal English sentence, construct by construct, for any domain:
    variables keep their own names. A reading on the position the innermost look-ahead reaches needs no words; one on an
    earlier position says which (`now`, `after the opponent's move`). A construct without a template is quoted as its
    source, so nothing is dropped."""

    def to_sentence(self, source: str) -> str:
        try:
            tree = ast.parse(source.strip(), mode="eval")
        except SyntaxError:
            return SOURCE_TEMPLATE.format(source=source.strip())
        return self._phrase(tree.body, ({HERE: NOW_WORD}, HERE), {})

    def _phrase(self, node: ast.expr, views: Views, places: dict[str, str]) -> str:
        match node:
            case ast.Constant(value=value):
                return self._value(value)
            case ast.Name(id=name) if name in places:
                return places[name]
            case ast.Name(id=name) if name in PLAYER_WORDS:
                return PLAYER_WORDS[name][1]
            case ast.Compare(left=left, ops=[operator], comparators=[right]):
                word = COMPARISON_WORDS.get(type(operator).__name__)
                if word is not None:
                    return f"{self._phrase(left, views, places)} {word} {self._operand(right, views, places)}"
            case ast.BoolOp(op=operator, values=values):
                return f" {CONNECTIVE_WORDS[type(operator).__name__]} ".join(
                    self._phrase(value, views, places) for value in values
                )
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                return NOT_TEMPLATE.format(body=self._phrase(operand, views, places))
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return NEGATIVE_TEMPLATE.format(body=self._phrase(operand, views, places))
            case ast.BinOp():
                return self._arithmetic(node, views, places)
            case ast.Subscript(value=ast.Attribute(value=ast.Name(id=view), attr=base), slice=index) if view in views[0]:
                return self._qualified(self._indexed(base, index, places), view, views)
            case ast.Attribute(value=ast.Name(id=view), attr=base) if view in views[0]:
                return self._qualified(READING_TEMPLATE.format(base=base), view, views)
            case ast.Call():
                phrase = self._call(node, views, places)
                if phrase is not None:
                    return phrase
        return SOURCE_TEMPLATE.format(source=ast.unparse(node))

    def _call(self, node: ast.Call, views: Views, places: dict[str, str]) -> str | None:
        match node:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=view), attr=kind),
                args=[ast.Name(id=player), ast.Lambda(args=ast.arguments(args=[ast.arg(arg=after)]), body=body)],
            ) if view in views[0] and kind in (BEST, WORST, COUNT) and player in PLAYER_WORDS:
                possessive = PLAYER_WORDS[player][0]
                before = views[0][view]
                where = (
                    AFTER_MOVE.format(possessive=possessive)
                    if view == HERE
                    else THEN_MOVE.format(before=before, possessive=possessive)
                )
                inner: Views = ({**views[0], after: where}, after)
                phrase = self._phrase(body, inner, places)
                if kind != COUNT and isinstance(body, ast.Compare | ast.BoolOp):
                    phrase = WHETHER_TEMPLATE.format(body=phrase)
                return LOOK_AHEAD_TEMPLATES[kind].format(possessive=possessive, body=phrase)
            case ast.Call(func=ast.Attribute(value=ast.Name(id=view), attr="mobility"), args=[ast.Name(id=player)]) if (
                view in views[0] and player in PLAYER_WORDS
            ):
                return self._qualified(MOBILITY_TEMPLATE.format(subject=PLAYER_WORDS[player][1]), view, views)
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id=view), attr="offset"),
                args=[ast.Constant(value=str() as base), ast.Name(id=place), *steps],
            ) if view in views[0] and place in places and all(isinstance(step, ast.Constant | ast.UnaryOp) for step in steps):
                phrase = OFFSET_TEMPLATE.format(
                    base=base, steps=", ".join(ast.unparse(step) for step in steps), place=places[place]
                )
                return self._qualified(phrase, view, views)
            case ast.Call(func=ast.Name(id="sum" | "min" | "max" as kind), args=[ast.GeneratorExp() as generator]):
                return self._aggregate(kind, generator, views, places)
            case ast.Call(func=ast.Name(id="max"), args=[ast.Constant(value=1), second]):
                return AT_LEAST_ONE_TEMPLATE.format(body=self._phrase(second, views, places))
            case ast.Call(func=ast.Name(id="max" | "min" as kind), args=[first, second]):
                template = LARGER_TEMPLATE if kind == "max" else SMALLER_TEMPLATE
                return template.format(first=self._phrase(first, views, places), second=self._phrase(second, views, places))
            case ast.Call(func=ast.Name(id="abs"), args=[ast.BinOp(op=ast.Sub(), left=first, right=second)]):
                return DISTANCE_TEMPLATE.format(
                    first=self._phrase(first, views, places), second=self._phrase(second, views, places)
                )
            case ast.Call(func=ast.Name(id="abs"), args=[body]):
                return SIZE_TEMPLATE.format(body=self._phrase(body, views, places))
            case ast.Call(func=ast.Name(id="wins"), args=[ast.Name(id=player), *_]) if player in PLAYER_WORDS:
                return WINS_TEMPLATE.format(subject=PLAYER_WORDS[player][1])
            case ast.Call(func=ast.Name(id="solo_distance"), args=[ast.Name(id=player), *_]) if player in PLAYER_WORDS:
                return SOLO_DISTANCE_TEMPLATE.format(possessive=PLAYER_WORDS[player][0])
            case ast.Call(func=ast.Name(id="win_chance")):
                return WIN_CHANCE_TEMPLATE
        return None

    def _aggregate(self, kind: str, generator: ast.GeneratorExp, views: Views, places: dict[str, str]) -> str | None:
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
            return PATTERN_TEMPLATE.format(body=self._conditions(conditions, views, {**places, PATTERN_INDEX: PLACE}))
        if targets == [AGGREGATE_INDEX]:
            entries, counted, inner = EVERY_ENTRY, ENTRIES, {**places, AGGREGATE_INDEX: THAT_ENTRY}
        elif targets == [AGGREGATE_INDEX, AGGREGATE_OTHER_INDEX]:
            entries, counted = EVERY_PAIR, PAIRS
            inner = {**places, AGGREGATE_INDEX: ONE_ENTRY, AGGREGATE_OTHER_INDEX: OTHER_ENTRY}
        else:
            return None
        if counting:
            return AGGREGATE_TEMPLATES["count"].format(entries=counted, body=self._conditions(conditions, views, inner))
        if conditions:
            return None
        return AGGREGATE_TEMPLATES[kind].format(entries=entries, body=self._phrase(generator.elt, views, inner))

    def _conditions(self, conditions: list[ast.expr], views: Views, places: dict[str, str]) -> str:
        return " and ".join(self._phrase(condition, views, places) for condition in conditions)

    def _arithmetic(self, node: ast.BinOp, views: Views, places: dict[str, str]) -> str:
        if isinstance(node.op, ast.Sub) and self._normalized(node.left, views) == self._normalized(node.right, views):
            return CHANGE_TEMPLATE.format(body=self._phrase(node.left, views, places))
        word = ARITHMETIC_WORDS.get(type(node.op).__name__)
        if word is None:
            return SOURCE_TEMPLATE.format(source=ast.unparse(node))
        return f"{self._part(node.left, views, places)} {word} {self._part(node.right, views, places)}"

    def _part(self, node: ast.expr, views: Views, places: dict[str, str]) -> str:
        phrase = self._phrase(node, views, places)
        return f"({phrase})" if isinstance(node, ast.BinOp) else phrase

    def _operand(self, node: ast.expr, views: Views, places: dict[str, str]) -> str:
        if isinstance(node, ast.Name) and node.id in PLAYER_VALUES:
            return PLAYER_VALUES[node.id]
        return self._phrase(node, views, places)

    def _indexed(self, base: str, index: ast.expr, places: dict[str, str]) -> str:
        match index:
            case ast.Name(id=place) if place == PATTERN_INDEX and place in places:
                return f"{READING_TEMPLATE.format(base=base)} {places[place]}"
            case ast.Name(id=place) if place in places:
                return OF_TEMPLATE.format(base=base, entry=places[place])
            case ast.Name(id=player) if player in PLAYER_WORDS:
                return OWNED_TEMPLATE.format(possessive=PLAYER_WORDS[player][0], base=base)
            case ast.Tuple(elts=elements) if all(isinstance(element, ast.Constant) for element in elements):
                return AT_TEMPLATE.format(base=base, indices=", ".join(self._value(element.value) for element in elements))  # type: ignore[attr-defined]
            case ast.Constant(value=value):
                return AT_TEMPLATE.format(base=base, indices=self._value(value))
        return f"{READING_TEMPLATE.format(base=base)} at `{ast.unparse(index)}`"

    def _qualified(self, reading: str, view: str, views: Views) -> str:
        if view == views[1]:
            return reading
        return QUALIFIED_TEMPLATE.format(reading=reading, view=views[0][view])

    def _normalized(self, node: ast.expr, views: Views) -> str:
        """The node with every view's name alike, so the same reading on two positions compares equal."""
        copy = ast.parse(ast.unparse(node), mode="eval")
        for child in ast.walk(copy):
            if isinstance(child, ast.Name) and child.id in views[0]:
                child.id = HERE
        return ast.dump(copy)

    def _value(self, value: object) -> str:
        if value is None:
            return EMPTY_WORD
        if value == OUTSIDE:
            return OUTSIDE_WORD
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
