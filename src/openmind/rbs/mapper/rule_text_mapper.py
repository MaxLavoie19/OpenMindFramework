from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.rbs.model.rule import Rule


class RuleTextMapper:
    """Maps a rule to readable text: place when cell(row,col) == None and cell(2,2) == None: EV 0.75 over 400 visits."""

    def __init__(self, expression_text_mapper: ExpressionTextMapper) -> None:
        self._expression_text_mapper = expression_text_mapper

    def to_text(self, rule: Rule) -> str:
        conditions = " and ".join(self._expression_text_mapper.to_text(condition) for condition in rule.conditions)
        scope = f"when {conditions}" if conditions else "in any state"
        return f"{rule.action} {scope}: EV {rule.expected_value} over {rule.visits} visits"
