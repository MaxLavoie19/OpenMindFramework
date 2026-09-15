from collections.abc import Sequence

from openmind.rbs.model.rule_explanation import RuleExplanation
from openmind.rbs.model.value_base import ValueBase


class ValueBaseExplanationMapper:
    """Maps a value base's explained rules to Markdown: a heading with the domain, then the bias and payoff range, then a
    table with a header row, one line per rule in the value base's order: Weight, Explanation (the language model's
    sentence, or "none" without one), Literal reading, Rule."""

    def to_markdown(self, value_base: ValueBase, explanations: Sequence[RuleExplanation]) -> str:
        models = sorted({explanation.model for explanation in explanations if explanation.model is not None})
        lines = [
            f"# {value_base.domain} value rules",
            "",
            f"{len(explanations)} rules; bias {value_base.bias:+.6g}; payoffs from {value_base.low:g} to "
            f"{value_base.high:g}; explained by {', '.join(models) if models else 'no language model'}.",
            "",
            "| Weight | Explanation | Literal reading | Rule |",
            "|---|---|---|---|",
        ]
        for explanation in explanations:
            sentence = "none" if explanation.sentence is None else self._cell(explanation.sentence)
            lines.append(
                f"| {explanation.weight:+.4g} | {sentence} | {self._cell(explanation.reading)} "
                f"| `{self._cell(explanation.source)}` |"
            )
        return "\n".join(lines) + "\n"

    def _cell(self, text: str) -> str:
        return " ".join(text.split()).replace("|", "\\|")
