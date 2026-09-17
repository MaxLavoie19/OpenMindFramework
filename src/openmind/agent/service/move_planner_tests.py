from openmind.agent.service.move_planner import MovePlanner
from openmind.mcts.constant.mcts_constant import FULL_OPTION, RANDOM_OPTION, SEARCH_OPTION


def test_the_fallback_runs_while_it_leaves_time_to_search_and_the_search_alone_otherwise() -> None:
    planner = MovePlanner()
    planner.observe("fallback", 1.0, 20)

    # 20 legal moves: the fallback costs 1.0 s
    assert planner.plan(1.5, 20, True, True) == FULL_OPTION
    assert planner.plan(1.0, 20, True, True) == SEARCH_OPTION
    assert planner.plan(0.01, 20, True, True) == SEARCH_OPTION


def test_a_fallback_never_measured_runs_and_an_agent_without_a_fallback_or_a_valuer_searches() -> None:
    assert MovePlanner().plan(0.01, 20, True, True) == FULL_OPTION
    assert MovePlanner().plan(5.0, 20, False, True) == SEARCH_OPTION
    assert MovePlanner().plan(5.0, 20, True, False) == SEARCH_OPTION


def test_only_a_budget_of_0_plays_a_random_move() -> None:
    assert MovePlanner().plan(0.0, 20, True, True) == RANDOM_OPTION
    assert MovePlanner().plan(-1.0, 20, False, False) == RANDOM_OPTION
