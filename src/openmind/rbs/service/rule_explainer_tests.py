from pathlib import Path

from openmind.rbs.factory.rbs_factory import create_rule_explainer
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.service.consequence_library_tests import strip_domain
from openmind.rule.model.python_rule import PythonRule

VALUE_BASE = ValueBase(
    "strip",
    0.1,
    0.0,
    1.0,
    (
        ValueRule(PythonRule("here.count(me, lambda v1: v1.payoff[me] == 1.0)"), 1.5),
        ValueRule(PythonRule("here.mobility(other)"), -0.25),
    ),
)


class FakeLanguageModel:
    """A language model answering from a list, None once it runs out, and keeping the prompts."""

    def __init__(self, *answers: str | None) -> None:
        self.name = "fake:1b"
        self.prompts: list[str] = []
        self._answers = list(answers)

    def complete(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self._answers.pop(0) if self._answers else None


def test_without_a_language_model_every_rule_gets_its_literal_reading_only() -> None:
    explanations = create_rule_explainer().explain(VALUE_BASE, strip_domain())

    assert [(explanation.weight, explanation.reading, explanation.sentence, explanation.model) for explanation in explanations] == [
        (1.5, "the number of my moves after which my payoff is 1", None, None),
        (-0.25, "the number of moves the opponent could make", None, None),
    ]


def test_a_language_model_is_asked_once_per_rule_and_its_sentences_are_cached(tmp_path: Path) -> None:
    explainer, domain = create_rule_explainer(), strip_domain()
    first = FakeLanguageModel("How many winning moves I have.", None)

    explanations = explainer.explain(VALUE_BASE, domain, first, tmp_path)  # type: ignore[arg-type]
    second = FakeLanguageModel("How few moves the opponent has.")
    again = explainer.explain(VALUE_BASE, domain, second, tmp_path)  # type: ignore[arg-type]

    assert [explanation.sentence for explanation in explanations] == ["How many winning moves I have.", None]
    assert [explanation.sentence for explanation in again] == ["How many winning moves I have.", "How few moves the opponent has."]
    assert (len(first.prompts), len(second.prompts)) == (2, 1)
    prompt = first.prompts[0]
    assert 'values positions of the game "strip"' in prompt
    assert "- `cell[1, 1]`, and at each of its 4 indices: None" in prompt
    assert "- `turn`: 'X'" in prompt
    assert "The rule's weight is +1.5" in prompt
    assert "Python: here.count(me, lambda v1: v1.payoff[me] == 1.0)" in prompt
    assert "Literal reading: the number of my moves after which my payoff is 1" in prompt
    assert (tmp_path / "strip" / "fake_1b.json").exists()
