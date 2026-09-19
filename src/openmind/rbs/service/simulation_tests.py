from collections.abc import Callable

import pytest

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system, create_simulation
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players

type Game = Callable[[str], RuleBasedGame]


def test_one_simulation_runs_any_game_s_rbs_given_to_it(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")
    game("rockpaperscissors")
    simulation = create_simulation()
    tictactoe, throwing = (create_rule_based_system(knowledge, name) for name in ("tictactoe", "rockpaperscissors"))

    assert simulation.players(tictactoe) == Players(("X", "O"), "payoff")
    assert len(simulation.actions(tictactoe, simulation.start(tictactoe))) == 9
    assert simulation.acting(throwing, simulation.start(throwing)) == (0, 1)


def test_the_player_acting_is_the_one_the_constraints_give_an_action(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")
    simulation, rbs = create_simulation(), create_rule_based_system(knowledge, "tictactoe")
    start = simulation.start(rbs)

    marked = simulation.outcomes(rbs, start, Action("place", (("col", 1), ("row", 1)))).outcomes[0][0]

    assert (simulation.acting_player(rbs, start), simulation.acting_player(rbs, marked)) == ("X", "O")
    assert simulation.actions(rbs, start, player="O") == ()


def test_players_acting_at_once_choose_joint_actions(game: Game, knowledge: KnowledgeBase) -> None:
    game("rockpaperscissors")
    simulation, rbs = create_simulation(), create_rule_based_system(knowledge, "rockpaperscissors")
    start = simulation.start(rbs)
    rock, paper = Action("throw", (("shape", "rock"),)), Action("throw", (("shape", "paper"),))

    ((after, _),) = simulation.joint_outcomes(rbs, start, JointAction((("A", rock), ("B", paper)))).outcomes

    assert [index for index, _ in simulation.joint_actions(rbs, start)] == [0, 1]
    with pytest.raises(ValueError, match="act at once"):
        simulation.actions(rbs, start)
    assert simulation.acting(rbs, after) == ()


def test_a_game_that_says_why_it_ended_says_so(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")
    simulation, rbs = create_simulation(), create_rule_based_system(knowledge, "tictactoe")

    assert simulation.ended(rbs, simulation.start(rbs)) is None
