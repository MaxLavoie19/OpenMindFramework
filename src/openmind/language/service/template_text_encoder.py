from openmind.language.constant import meaning_constant as meaning
from openmind.language.constant import template_constant as words
from openmind.language.model.meaning import Meaning


class TemplateTextEncoder:
    """Turns a meaning into a literal English reading, node by node, with the templates of template_constant: exact
    and stiff. A rule's meaning starts with its effect (`Raises my value: ...`); a measure alone reads without it. A
    node without a template is quoted as JSON-like text, so nothing is dropped."""

    @property
    def name(self) -> str:
        return words.TEMPLATE_ENCODER

    def encode(self, node: Meaning) -> str:
        if meaning.EFFECT in node and meaning.MEASURE in node:
            return words.RULE_TEMPLATES[str(node[meaning.EFFECT])].format(measure=self._text(node[meaning.MEASURE]))  # type: ignore[arg-type]
        return self._text(node)

    def _text(self, node: Meaning) -> str:
        kind = next(iter(node), None)
        match kind:
            case meaning.NUMBER | meaning.NAME:
                return str(node[kind])
            case meaning.EMPTY:
                return words.EMPTY_WORD
            case meaning.OUTSIDE:
                return words.OUTSIDE_WORD
            case meaning.TRUTH:
                return str(node[kind]).lower()
            case meaning.PLAYER:
                return words.SUBJECTS[str(node[kind])]
            case meaning.READING:
                return self._qualified(self._reading(node), node)
            case meaning.MOVES_AVAILABLE_TO:
                return self._qualified(words.MOBILITY_TEMPLATE.format(subject=words.SUBJECTS[str(node[kind])]), node)
            case meaning.LOOK_AHEAD:
                body: Meaning = node[meaning.OF]  # type: ignore[assignment]
                text = self._text(body)
                if node[kind] != meaning.COUNT and next(iter(body)) in (meaning.COMPARE, meaning.ALL_OF, meaning.ANY_OF):
                    text = words.WHETHER_TEMPLATE.format(body=text)
                template = words.LOOK_AHEAD_TEMPLATES[str(node[kind])]
                return template.format(possessive=words.POSSESSIVES[str(node[meaning.MOVES_OF])], body=text)
            case meaning.COMPARE:
                right: Meaning = node[meaning.RIGHT]  # type: ignore[assignment]
                operand = (
                    words.OWNERSHIP[str(right[meaning.PLAYER])] if next(iter(right)) == meaning.PLAYER else self._text(right)
                )
                return f"{self._text(node[meaning.LEFT])} {words.COMPARISON_WORDS[str(node[kind])]} {operand}"  # type: ignore[arg-type]
            case meaning.ARITHMETIC:
                return f"{self._part(node[meaning.LEFT])} {node[kind]} {self._part(node[meaning.RIGHT])}"  # type: ignore[arg-type]
            case meaning.CHANGE_IN:
                return words.CHANGE_TEMPLATE.format(body=self._text(node[kind]))  # type: ignore[arg-type]
            case meaning.DISTANCE_BETWEEN | meaning.LARGER_OF | meaning.SMALLER_OF:
                first, second = node[kind]  # type: ignore[misc]
                template = {
                    meaning.DISTANCE_BETWEEN: words.DISTANCE_TEMPLATE,
                    meaning.LARGER_OF: words.LARGER_TEMPLATE,
                    meaning.SMALLER_OF: words.SMALLER_TEMPLATE,
                }[kind]
                return template.format(first=self._text(first), second=self._text(second))
            case meaning.SIZE_OF:
                return words.SIZE_TEMPLATE.format(body=self._text(node[kind]))  # type: ignore[arg-type]
            case meaning.AT_LEAST_ONE:
                return words.AT_LEAST_ONE_TEMPLATE.format(body=self._text(node[kind]))  # type: ignore[arg-type]
            case meaning.ALL_OF | meaning.ANY_OF:
                return f" {words.CONNECTIVE_WORDS[kind]} ".join(self._text(part) for part in node[kind])  # type: ignore[attr-defined]
            case meaning.NOT:
                return words.NOT_TEMPLATE.format(body=self._text(node[kind]))  # type: ignore[arg-type]
            case meaning.NEGATIVE:
                return words.NEGATIVE_TEMPLATE.format(body=self._text(node[kind]))  # type: ignore[arg-type]
            case meaning.COUNT if node[kind] == meaning.PLACES:
                return words.PLACES_TEMPLATE.format(body=self._conditions(node[meaning.WHERE]))  # type: ignore[arg-type]
            case meaning.AGGREGATE:
                over = str(node[meaning.OVER])
                if node[kind] == meaning.COUNT:
                    return words.AGGREGATE_TEMPLATES["count"].format(entries=over, body=self._conditions(node[meaning.WHERE]))  # type: ignore[arg-type]
                return words.AGGREGATE_TEMPLATES[str(node[kind])].format(
                    every=words.EVERY_WORDS[over], body=self._text(node[meaning.OF])  # type: ignore[arg-type]
                )
            case meaning.WINNING_MOVES_OF:
                return words.WINS_TEMPLATE.format(subject=words.SUBJECTS[str(node[kind])])
            case meaning.WIN_CHANCE_OF_THE_ACTION:
                return words.WIN_CHANCE_TEMPLATE
            case meaning.SOURCE:
                return words.SOURCE_TEMPLATE.format(source=node[kind])
        return words.SOURCE_TEMPLATE.format(source=node)

    def _reading(self, node: Meaning) -> str:
        base = str(node[meaning.READING])
        if meaning.STEPS in node:
            steps = ", ".join(str(step) for step in node[meaning.STEPS])  # type: ignore[attr-defined]
            place = self._index_word(node[meaning.FROM])  # type: ignore[arg-type]
            return words.OFFSET_TEMPLATE.format(base=base, steps=steps, place=place)
        if meaning.AT not in node:
            return words.READING_TEMPLATE.format(base=base)
        index: Meaning = node[meaning.AT]  # type: ignore[assignment]
        kind = next(iter(index))
        match kind:
            case meaning.PLACE:
                return words.PLACE_READING_TEMPLATE.format(base=base, place=words.PLACE_WORDS[str(index[kind])])
            case meaning.ENTRY:
                return words.OF_TEMPLATE.format(base=base, entry=words.ENTRY_WORDS[str(index[kind])])
            case meaning.PLAYER:
                return words.OWNED_TEMPLATE.format(possessive=words.POSSESSIVES[str(index[kind])], base=base)
            case meaning.LABEL:
                return words.AT_TEMPLATE.format(base=base, indices=index[kind])
            case meaning.INDICES:
                return words.AT_TEMPLATE.format(base=base, indices=", ".join(self._index_value(value) for value in index[kind]))  # type: ignore[attr-defined]
        return words.SOURCE_INDEX_TEMPLATE.format(base=base, source=index.get(meaning.SOURCE, index))

    def _index_word(self, index: Meaning) -> str:
        kind = next(iter(index))
        if kind == meaning.PLACE:
            return words.PLACE_WORDS[str(index[kind])]
        if kind == meaning.ENTRY:
            return words.ENTRY_WORDS[str(index[kind])]
        return self._text(index)

    def _index_value(self, value: object) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _qualified(self, text: str, node: Meaning) -> str:
        position = node.get(meaning.POSITION)
        if not isinstance(position, dict):
            return text
        if position.get(meaning.NOW):
            return words.QUALIFIED_TEMPLATE.format(reading=text, position=words.NOW_WORD)
        moved = list(position.get(meaning.AFTER_MOVES_OF, []))
        described = words.AFTER_MOVE.format(possessive=words.POSSESSIVES[moved[0]])
        for player in moved[1:]:
            described = words.THEN_MOVE.format(before=described, possessive=words.POSSESSIVES[player])
        return words.QUALIFIED_TEMPLATE.format(reading=text, position=described)

    def _conditions(self, conditions: list[Meaning]) -> str:
        return " and ".join(self._text(condition) for condition in conditions)

    def _part(self, node: Meaning) -> str:
        text = self._text(node)
        return f"({text})" if next(iter(node)) == meaning.ARITHMETIC else text
