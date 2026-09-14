import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action
from openmind.world.model.state import State

CENTER = Action("place", (("col", 2), ("row", 2)))


class FavourCenter:
    """A rater rating the center 1.0 and every other action 0.0."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple(1.0 if action == CENTER else 0.0 for action in actions)


@pytest.mark.log_level("INFO")
def test_build_gives_an_agent_that_searches() -> None:
    domain = create_tictactoe_domain()
    builder = StateBuilder()
    x_can_win = {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"}
    for name, value in (dict(domain.initial_state.variables) | x_can_win).items():
        builder.with_variable(name, value)

    agent = AgentBuilder().with_iterations(100).with_exploration(1.4).with_seed(1).build()

    assert agent.choose(domain, builder.build()) == Action("place", (("col", 3), ("row", 1)))


@pytest.mark.log_level("INFO")
def test_build_with_guidance_expands_the_best_rated_action_first() -> None:
    domain = create_tictactoe_domain()

    agent = AgentBuilder().with_iterations(1).with_exploration(1.4).with_seed(1).with_guidance(FavourCenter()).build()

    assert agent.choose(domain, domain.initial_state) == CENTER


class CountingCenter(FavourCenter):
    """FavourCenter, counting how often it is asked to rate."""

    def __init__(self) -> None:
        self.calls = 0

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        self.calls += 1
        return super().rate(state, actions)


@pytest.mark.log_level("INFO")
def test_build_without_guided_rollouts_rates_only_the_tree_nodes() -> None:
    domain = create_tictactoe_domain()
    guided, unguided = CountingCenter(), CountingCenter()

    for rater, rollouts in ((guided, True), (unguided, False)):
        builder = AgentBuilder().with_iterations(1).with_exploration(1.4).with_seed(1).with_guidance(rater)
        builder.with_guided_rollouts(rollouts).build().choose(domain, domain.initial_state)

    # One iteration creates the root and the node after the center; only a guided rollout rates its steps too.
    assert unguided.calls == 2
    assert guided.calls > 2


class ValuesEverything:
    """A valuer giving every position a draw, counting how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def value(self, state: State) -> tuple[float, ...] | None:
        self.calls += 1
        return (0.5, 0.5)


@pytest.mark.log_level("INFO")
def test_build_with_valuation_values_positions_instead_of_playing_them_out() -> None:
    domain = create_tictactoe_domain()
    valuer = ValuesEverything()

    AgentBuilder().with_iterations(3).with_exploration(1.4).with_seed(1).with_valuation(valuer).build().choose(
        domain, domain.initial_state
    )

    # Each iteration reaches one new position still in play, and values it.
    assert valuer.calls == 3


@pytest.mark.log_level("INFO")
def test_build_with_a_rollout_limit_gives_rollouts_still_in_play_the_unfinished_payoff() -> None:
    domain = create_tictactoe_domain()

    agent = AgentBuilder().with_iterations(9).with_exploration(1.4).with_seed(1).with_rollout_limit(0, 0.25).build()

    # Each iteration tries a new first move, and its rollout stops at once.
    assert {(item.visits, item.mean_payoff) for item in agent.search(domain, domain.initial_state).statistics} == {(1, 0.25)}


@pytest.mark.parametrize(("limit", "payoff", "message"), [(-1, 0.5, "negative"), (5, None, "unfinished payoff")])
def test_build_rejects_a_negative_rollout_limit_or_one_without_an_unfinished_payoff(
    limit: int, payoff: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_rollout_limit(limit, payoff).build()


def test_build_rejects_negative_rollout_actions() -> None:
    with pytest.raises(ValueError, match="-1"):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_rollout_actions(-1).build()


def test_build_rejects_missing_settings() -> None:
    with pytest.raises(ValueError, match="iterations"):
        AgentBuilder().with_exploration(1.4).build()


def test_build_rejects_fewer_than_one_iteration() -> None:
    with pytest.raises(ValueError, match="0"):
        AgentBuilder().with_iterations(0).with_exploration(1.4).build()
