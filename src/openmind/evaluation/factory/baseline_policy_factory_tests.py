import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.evaluation.factory.baseline_policy_factory import (
    create_built_agent,
    create_random_policy,
    create_seeded_agent,
)

pytestmark = pytest.mark.log_level("INFO")


def test_a_random_policy_draws_its_choices_from_the_game_s_seed() -> None:
    domain = create_tictactoe_domain()

    assert len({create_random_policy(7).choose(domain, domain.initial_state) for _ in range(3)}) == 1
    assert len({create_random_policy(seed).choose(domain, domain.initial_state) for seed in range(20)}) > 1


def test_a_built_agent_keeps_the_builder_s_seed_whatever_the_game_s() -> None:
    domain = create_tictactoe_domain()
    builder = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(3)

    assert create_built_agent(builder, 1).search(domain, domain.initial_state) == create_built_agent(builder, 2).search(
        domain, domain.initial_state
    )


def test_a_seeded_agent_searches_with_the_game_s_seed() -> None:
    domain = create_tictactoe_domain()
    builder = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(3)

    first, again, other = (create_seeded_agent(builder, seed).search(domain, domain.initial_state) for seed in (1, 1, 2))

    assert first == again
    assert first != other
