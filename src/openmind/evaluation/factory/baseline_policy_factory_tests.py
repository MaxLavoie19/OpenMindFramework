from collections.abc import Callable

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.evaluation.factory.baseline_policy_factory import (
    create_built_agent,
    create_random_policy,
    create_seeded_agent,
)

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]


def test_a_random_policy_draws_its_choices_from_the_game_s_seed(game: Game) -> None:
    rbs = game("tictactoe")

    assert len({create_random_policy(7).choose(rbs, rbs.start()) for _ in range(3)}) == 1
    assert len({create_random_policy(seed).choose(rbs, rbs.start()) for seed in range(20)}) > 1


def test_a_built_agent_keeps_the_builder_s_seed_whatever_the_game_s(game: Game) -> None:
    rbs = game("tictactoe")
    builder = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(3)

    assert create_built_agent(builder, 1).search(rbs, rbs.start()) == create_built_agent(builder, 2).search(
        rbs, rbs.start()
    )


def test_a_seeded_agent_searches_with_the_game_s_seed(game: Game) -> None:
    rbs = game("tictactoe")
    builder = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(3)

    first, again, other = (create_seeded_agent(builder, seed).search(rbs, rbs.start()) for seed in (1, 1, 2))

    assert first == again
    assert first != other
