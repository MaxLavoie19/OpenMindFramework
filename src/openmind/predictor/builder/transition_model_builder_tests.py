import pytest

from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel


def test_build_keeps_transitions_in_order() -> None:
    roll = (Branch(0.5, ()), Branch(0.5, ()))
    certain = (Branch(1.0, ()),)

    model = TransitionModelBuilder().with_transition("roll", roll).with_transition("pass", certain).build()

    assert model == TransitionModel((Transition("roll", roll), Transition("pass", certain)))


def test_with_transition_accepts_probabilities_that_sum_to_one_after_rounding() -> None:
    faces = tuple(Branch(1 / 6, ()) for _ in range(6))

    model = TransitionModelBuilder().with_transition("roll", faces).build()

    assert model.transitions[0].branches == faces


def test_with_transition_rejects_an_action_that_already_has_one() -> None:
    builder = TransitionModelBuilder().with_transition("pass", (Branch(1.0, ()),))

    with pytest.raises(ValueError, match="pass"):
        builder.with_transition("pass", (Branch(1.0, ()),))


@pytest.mark.parametrize("probabilities", [(), (0.5,), (0.7, 0.7)])
def test_with_transition_rejects_probabilities_not_summing_to_one(probabilities: tuple[float, ...]) -> None:
    branches = tuple(Branch(probability, ()) for probability in probabilities)

    with pytest.raises(ValueError, match="roll"):
        TransitionModelBuilder().with_transition("roll", branches)
