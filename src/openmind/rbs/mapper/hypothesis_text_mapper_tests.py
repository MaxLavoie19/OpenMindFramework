from openmind.rbs.mapper.hypothesis_text_mapper import HypothesisTextMapper
from openmind.rbs.model.hypothesis_test import HypothesisTest
from openmind.rule.model.python_rule import PythonRule


def test_a_validated_hypothesis() -> None:
    test = HypothesisTest("place", (PythonRule("win_chance(action) >= 1"),), 1, 0.61, 40, 0.58, 12, 0.0001, 0.002, True)

    assert HypothesisTextMapper().to_text(test) == (
        "place when win_chance(action) >= 1: raises advantage, discovery 0.61 over 40 states, "
        "validation 0.58 over 12 states, p 0.0001, q 0.002, validated"
    )


def test_a_rejected_hypothesis_without_validation_states() -> None:
    test = HypothesisTest("place", (PythonRule("col == 4"),), -1, -0.2, 6, None, 0, 1.0, 1.0, False)

    assert HypothesisTextMapper().to_text(test) == (
        "place when col == 4: lowers advantage, discovery -0.2 over 6 states, validation none over 0 states, "
        "p 1, q 1, rejected"
    )
