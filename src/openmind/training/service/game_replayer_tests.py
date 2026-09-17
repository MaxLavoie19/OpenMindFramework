from dataclasses import replace

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.model_description import ModelDescription
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.training.mapper.played_game_summary_mapper import PlayedGameSummaryMapper
from openmind.training.service.game_replayer import GameReplayer
from openmind.training.service.self_play_tests import new_self_play


def test_a_replayed_game_has_every_position_the_game_had() -> None:
    domain = create_tictactoe_domain()
    builders = (AgentBuilder().with_iterations(10).with_exploration(1.4), AgentBuilder().with_iterations(10).with_exploration(1.4))
    game = new_self_play().play_arm_game(domain, builders, ("a", "b"), 3, 4)
    models = (ModelDescription("a", "{}"), ModelDescription("b", "{}"))
    summary = PlayedGameSummaryMapper().to_summary(domain, game, "arms", None, 1, models, None)

    replayed = GameReplayer(create_predictor()).replay(domain, summary)

    assert replayed.states == game.states
    assert (replayed.payoffs, replayed.arms, replayed.actions) == (game.payoffs, ("a", "b"), game.actions)


def test_a_game_without_an_outcome_seed_can_t_be_replayed() -> None:
    domain = create_tictactoe_domain()
    game = new_self_play().play_game(domain, AgentBuilder().with_iterations(5).with_exploration(1.4), 1, 2)
    summary = PlayedGameSummaryMapper().to_summary(domain, game, "arms", None, 1, (ModelDescription("a", "{}"),) * 2, None)
    stripped = replace(summary, seeds=())

    with pytest.raises(ValueError, match="no outcome seed"):
        GameReplayer(create_predictor()).replay(domain, stripped)
