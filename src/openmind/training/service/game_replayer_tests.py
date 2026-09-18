from collections.abc import Callable
from dataclasses import replace

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.model_description import ModelDescription
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.service.game_replayer import GameReplayer
from openmind.training.service.self_play_tests import new_self_play


type Game = Callable[[str], RuleBasedSystem]


def test_a_replayed_game_has_every_position_the_game_had(game: Game) -> None:
    rbs = game("tictactoe")
    builders = (AgentBuilder().with_iterations(10).with_exploration(1.4), AgentBuilder().with_iterations(10).with_exploration(1.4))
    game = new_self_play().play_arm_game(rbs, builders, ("a", "b"), 3, 4)
    models = (ModelDescription("a", "{}"), ModelDescription("b", "{}"))
    summary = PlayedGameSummaryMapper().to_summary(rbs, game, "arms", None, 1, models, None)

    replayed = GameReplayer().replay(rbs, summary)

    assert replayed.states == game.states
    assert (replayed.payoffs, replayed.arms, replayed.actions) == (game.payoffs, ("a", "b"), game.actions)


def test_a_game_without_an_outcome_seed_can_t_be_replayed(game: Game) -> None:
    rbs = game("tictactoe")
    game = new_self_play().play_game(rbs, AgentBuilder().with_iterations(5).with_exploration(1.4), 1, 2)
    summary = PlayedGameSummaryMapper().to_summary(rbs, game, "arms", None, 1, (ModelDescription("a", "{}"),) * 2, None)
    stripped = replace(summary, seeds=())

    with pytest.raises(ValueError, match="no outcome seed"):
        GameReplayer().replay(rbs, stripped)
