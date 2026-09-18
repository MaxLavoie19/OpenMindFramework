from collections.abc import Callable
import pytest

from openmind.rbs.service.rule_based_system import RuleBasedSystem

from openmind.mcts.factory.move_prior_factory import create_move_prior

type Game = Callable[[str], RuleBasedSystem]


@pytest.mark.parametrize(("kind", "message"), [("rater", "rules that rate"), ("value", "value rules"), ("guess", "not 'guess'")])
def test_a_prior_without_the_model_it_reads_or_with_an_unknown_name_raises(game: Game, kind: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        create_move_prior(kind, 0.1, game("tictactoe"))
