import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.mcts.factory.move_prior_factory import create_move_prior


@pytest.mark.parametrize(("kind", "message"), [("rater", "rules that rate"), ("value", "value rules"), ("guess", "not 'guess'")])
def test_a_prior_without_the_model_it_reads_or_with_an_unknown_name_raises(kind: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_move_prior(kind, 0.1, create_tictactoe_domain())
