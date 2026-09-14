import pytest

from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule

NOTHING = PythonRule("")


def test_build_keeps_transitions_in_order_and_the_definitions() -> None:
    roll = (Branch(0.5, PythonRule("score = 1")), Branch(0.5, NOTHING))
    certain = (Branch(1.0, NOTHING),)
    definitions = PythonRule("SIDES = 2")

    model = (
        TransitionModelBuilder()
        .with_transition("roll", roll)
        .with_transition("pass", certain)
        .with_definitions(definitions)
        .build()
    )

    assert model == TransitionModel((Transition("roll", roll), Transition("pass", certain)), definitions)


def test_with_transition_accepts_probabilities_that_sum_to_one_after_rounding() -> None:
    faces = tuple(Branch(1 / 6, PythonRule(f"face = {face}")) for face in range(1, 7))

    model = TransitionModelBuilder().with_transition("roll", faces).build()

    assert (model.transitions[0].branches, model.definitions) == (faces, None)


def test_with_transition_rejects_an_action_that_already_has_one() -> None:
    builder = TransitionModelBuilder().with_transition("pass", (Branch(1.0, NOTHING),))

    with pytest.raises(ValueError, match="pass"):
        builder.with_transition("pass", (Branch(1.0, NOTHING),))


@pytest.mark.parametrize("probabilities", [(), (0.5,), (0.7, 0.7)])
def test_with_transition_rejects_probabilities_not_summing_to_one(probabilities: tuple[float, ...]) -> None:
    branches = tuple(Branch(probability, NOTHING) for probability in probabilities)

    with pytest.raises(ValueError, match="roll"):
        TransitionModelBuilder().with_transition("roll", branches)
