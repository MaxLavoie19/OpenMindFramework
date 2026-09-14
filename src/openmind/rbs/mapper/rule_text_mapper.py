from openmind.rbs.model.rule import Rule


class RuleTextMapper:
    """Maps a rule to readable text: place when cell[row, col] == None and cell[2, 2] == None: EV 0.75 over 400
    visits, with ", priority" added for a priority rule."""

    def to_text(self, rule: Rule) -> str:
        conditions = " and ".join(condition.source for condition in rule.conditions)
        scope = f"when {conditions}" if conditions else "in any state"
        priority = ", priority" if rule.priority else ""
        return f"{rule.action} {scope}: EV {rule.expected_value} over {rule.visits} visits{priority}"
