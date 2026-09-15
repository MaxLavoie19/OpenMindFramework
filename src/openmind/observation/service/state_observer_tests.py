import pytest

from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.observation.model.observation import Observation
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.state import State

STATE = State((("hand(A)", 1), ("hand(B)", 2), ("turn", "A")))
DEFINITIONS = PythonRule("CARDS = (1, 2, 3)\n\ndef other(player):\n    return 'B' if player == 'A' else 'A'")


def cards(completions: str = "[({'hand(' + other(player) + ')': card}, 0.5) for card in CARDS if card != hand[player]]") -> Observation:
    """Each player sees their own card, not the other's, which is any card but their own."""
    return Observation(PythonRule("('hand(' + other(player) + ')',)"), PythonRule(completions), DEFINITIONS)


def test_a_player_sees_their_own_variables_and_the_others_hidden() -> None:
    observer = create_state_observer()

    assert observer.observe(cards(), STATE, "A") == State((("hand(A)", 1), ("hand(B)", "<hidden>"), ("turn", "A")))
    assert observer.observe(cards(), STATE, "B") == State((("hand(A)", "<hidden>"), ("hand(B)", 2), ("turn", "A")))


def test_completions_fill_the_hidden_variables_with_each_possibility_and_its_probability() -> None:
    observer = create_state_observer()

    completions = observer.completions(cards(), observer.observe(cards(), STATE, "A"), "A")

    assert completions == (
        (State((("hand(A)", 1), ("hand(B)", 2), ("turn", "A"))), 0.5),
        (State((("hand(A)", 1), ("hand(B)", 3), ("turn", "A"))), 0.5),
    )


def test_completions_whose_probabilities_don_t_sum_to_1_raise() -> None:
    uneven = cards("[({'hand(B)': 2}, 0.5)]")

    with pytest.raises(ValueError, match="summing to 0.5"):
        create_state_observer().completions(uneven, STATE, "A")


def test_a_completion_that_doesn_t_give_exactly_the_hidden_variables_raises() -> None:
    too_much = cards("[({'hand(B)': 2, 'turn': 'B'}, 1.0)]")

    with pytest.raises(ValueError, match="not the hidden variables"):
        create_state_observer().completions(too_much, STATE, "A")


def test_hiding_a_variable_the_state_doesn_t_have_raises() -> None:
    with pytest.raises(KeyError, match="hand"):
        create_state_observer().observe(cards(), State((("hand(A)", 1), ("turn", "A"))), "A")
