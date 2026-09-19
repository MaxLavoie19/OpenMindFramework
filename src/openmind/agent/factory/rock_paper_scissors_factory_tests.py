from collections.abc import Callable
import pytest

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.service.state_reader import StateReader


type Game = Callable[[str], RuleBasedGame]


def throw(shape: str) -> Action:
    return Action("throw", (("shape", shape),))


def test_both_players_act_at_once_from_the_start(game: Game) -> None:
    rbs = game("rockpaperscissors")

    assert rbs.acting(rbs.start()) == (0, 1)


@pytest.mark.parametrize(
    ("first", "second", "payoffs"),
    [("rock", "scissors", (1.0, 0.0)), ("rock", "rock", (0.5, 0.5)), ("rock", "paper", (0.0, 1.0)), ("scissors", "paper", (1.0, 0.0))],
)
def test_the_hands_thrown_at_once_decide_the_payoffs_and_end_the_game(game: Game, 
    first: str, second: str, payoffs: tuple[float, float]
) -> None:
    rbs = game("rockpaperscissors")
    joint = JointAction((("A", throw(first)), ("B", throw(second))))

    ((state, probability),) = rbs.joint_outcomes(rbs.start(), joint).outcomes

    assert probability == 1.0
    assert state.model("hand") == Map.of({"A": first, "B": second})
    assert StateReader().payoffs(state, rbs.players()) == payoffs
    assert rbs.acting(state) == ()
