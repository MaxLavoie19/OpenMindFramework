from collections.abc import Callable
import pickle

import pytest

from openmind.evaluation.service.exact_search import ExactSearch
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


type Declare = Callable[..., RuleBasedSystem]


def new_exact_search() -> ExactSearch:
    return ExactSearch(StateReader())


def one_move(declared: Declare, **outcomes: object) -> RuleBasedSystem:
    """One player; every action is legal until the payoff is set, and every action sets it."""
    no_payoff = PythonRule("payoff is None")
    return declared(
        State((("payoff", None), ("turn", "me"))),
        legal={action: (no_payoff,) for action in outcomes},
        outcomes=outcomes,
        context="one move",
    )


def pay(payoff: float) -> PythonRule:
    return PythonRule(f"payoff = {payoff!r}")


def trust(declared: Declare) -> RuleBasedSystem:
    """A plays safe (0.5 each) or risky; after risky, B plays punish (A 0, B 1) or reward (A 1, B 0)."""
    unset = PythonRule("payoff['A'] is None")

    def payoffs(a: float, b: float) -> PythonRule:
        return PythonRule(f"payoff['A'] = {a!r}\npayoff['B'] = {b!r}")

    a_moves, b_moves = (unset, PythonRule("turn == 'A'")), (unset, PythonRule("turn == 'B'"))
    return declared(
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A"))),
        legal={"safe": a_moves, "risky": a_moves, "punish": b_moves, "reward": b_moves},
        outcomes={
            "safe": ((1.0, payoffs(0.5, 0.5)),),
            "risky": ((1.0, PythonRule("turn = 'B'")),),
            "punish": ((1.0, payoffs(0.0, 1.0)),),
            "reward": ((1.0, payoffs(1.0, 0.0)),),
        },
        players=Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
        context="trust",
    )


def test_action_values_give_each_legal_action_its_value_for_the_player_to_act(declared: Declare) -> None:
    rbs = trust(declared)
    after_risky = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))

    assert new_exact_search().action_values(rbs, rbs.start()) == (
        (Action("safe", ()), 0.5),
        (Action("risky", ()), 0.0),
    )
    assert new_exact_search().action_values(rbs, after_risky) == (
        (Action("punish", ()), 1.0),
        (Action("reward", ()), 0.0),
    )


def test_value_gives_every_player_payoff_under_perfect_play(declared: Declare) -> None:
    rbs = trust(declared)
    finished = State((("payoff(A)", 1.0), ("payoff(B)", 0.0), ("turn", "B")))

    assert new_exact_search().value(rbs, rbs.start()) == (0.5, 0.5)
    assert new_exact_search().value(rbs, finished) == (1.0, 0.0)


def test_the_winning_action_is_optimal(declared: Declare) -> None:
    rbs = one_move(declared, lose=((1.0, pay(0.0)),), win=((1.0, pay(1.0)),))

    assert new_exact_search().optimal_actions(rbs, rbs.start()) == (Action("win", ()),)


def test_each_player_maximizes_their_own_payoff(declared: Declare) -> None:
    rbs = trust(declared)
    after_risky = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))

    assert new_exact_search().optimal_actions(rbs, rbs.start()) == (Action("safe", ()),)
    assert new_exact_search().optimal_actions(rbs, after_risky) == (Action("punish", ()),)


def test_outcomes_count_by_their_probability(declared: Declare) -> None:
    gamble = ((0.5, pay(1.0)), (0.5, pay(0.0)))
    better = one_move(declared, gamble=gamble, safe=((1.0, pay(0.6)),))

    assert new_exact_search().optimal_actions(better, better.start()) == (Action("safe", ()),)


def test_positions_are_the_reachable_states_with_a_legal_action(declared: Declare) -> None:
    rbs = trust(declared)

    assert new_exact_search().positions(rbs) == (
        rbs.start(),
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B"))),
    )


def test_a_copy_sent_to_another_process_leaves_its_values_behind(declared: Declare) -> None:
    search, rbs = new_exact_search(), one_move(declared, win=((1.0, pay(1.0)),))
    values = search.action_values(rbs, rbs.start())

    copy = pickle.loads(pickle.dumps(search))

    assert copy.action_values(rbs, rbs.start()) == values


def test_a_state_without_legal_action_raises(declared: Declare) -> None:
    rbs = one_move(declared, win=((1.0, pay(1.0)),))

    with pytest.raises(ValueError, match="No legal action"):
        new_exact_search().optimal_actions(rbs, State((("payoff", 1.0), ("turn", "me"))))
