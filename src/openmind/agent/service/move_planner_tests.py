from openmind.agent.service.move_planner import MovePlanner
from openmind.mcts.constant.mcts_constant import FULL_OPTION, ONE_PLY_OPTION, RANDOM_OPTION, SEARCH_OPTION


def planner() -> MovePlanner:
    """Costs seen so far: a search iteration 0.1 s, the fallback 0.05 s a legal move, a one-ply valuation 0.02 s."""
    planned = MovePlanner()
    planned.observe("iteration", 5.0, 50)
    planned.observe("fallback", 1.0, 20)
    planned.observe("valuation", 0.4, 20)
    return planned


def test_the_richest_option_whose_cost_fits_the_budget_is_picked_down_to_a_random_move() -> None:
    # 20 legal moves: fallback and search 1.0 + 2.0 s, search 2.0 s, one-ply 0.4 s
    assert planner().plan(3.5, 20, True, True) == FULL_OPTION
    assert planner().plan(2.5, 20, True, True) == SEARCH_OPTION
    assert planner().plan(1.0, 20, True, True) == ONE_PLY_OPTION
    assert planner().plan(0.3, 20, True, True) == RANDOM_OPTION
    assert planner().plan(0.0, 20, True, True) == RANDOM_OPTION


def test_an_agent_without_a_fallback_or_a_valuer_skips_the_options_it_can_t_take() -> None:
    assert planner().plan(3.5, 20, False, True) == SEARCH_OPTION
    assert planner().plan(1.0, 20, True, False) == RANDOM_OPTION


def test_a_cost_never_measured_counts_as_fitting_and_costs_are_the_game_s_means() -> None:
    fresh = MovePlanner()
    assert fresh.plan(0.01, 20, True, True) == FULL_OPTION

    fresh.observe("iteration", 1.0, 10)
    fresh.observe("iteration", 3.0, 10)

    # the mean of 0.1 and 0.3 seconds an iteration, 0.2, for 20 legal moves
    assert fresh.plan(4.0, 20, False, False) == SEARCH_OPTION and fresh.plan(3.9, 20, False, False) == RANDOM_OPTION
