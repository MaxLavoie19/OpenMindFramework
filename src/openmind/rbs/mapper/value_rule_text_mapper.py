from openmind.rbs.model.value_rule import ValueRule


class ValueRuleTextMapper:
    """Maps a value rule to readable text, its weight then its term: +0.42 × wins(me)."""

    def to_text(self, rule: ValueRule) -> str:
        return f"{rule.weight:+.6g} × {rule.term.source}"
