from openmind.rbs.mapper.position_rule_explanation_mapper import PositionRuleExplanationMapper
from openmind.rbs.model.rule_explanation import RuleExplanation


def test_explained_rules_become_a_table_with_a_header_row() -> None:
    explanations = (
        RuleExplanation("(here.x[me]) | 1", 0.5, "`here.x[me] | 1`", "A rule with a | in it,\nover two lines.", "qwen3:8b"),
        RuleExplanation("here.mobility(me)", -1.25, "the number of moves I could make", None, "qwen3:8b"),
    )

    assert PositionRuleExplanationMapper().to_markdown("chess", explanations) == (
        "# chess position rules\n"
        "\n"
        "2 rules; explained by qwen3:8b.\n"
        "\n"
        "| Weight | Explanation | Literal reading | Rule |\n"
        "|---|---|---|---|\n"
        "| +0.5 | A rule with a \\| in it, over two lines. | `here.x[me] \\| 1` | `(here.x[me]) \\| 1` |\n"
        "| -1.25 | none | the number of moves I could make | `here.mobility(me)` |\n"
    )
