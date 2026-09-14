import pickle

import pytest

from openmind.agent.model.domain import Domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


def new_exact_search() -> ExactSearch:
    return ExactSearch(create_solver(), create_predictor(), StateReader())


def one_move(*transitions: Transition) -> Domain:
    """One player; every action is legal until the payoff is set, and every action sets it."""
    no_payoff = PythonRule("payoff is None")
    return Domain(
        "one move",
        State((("payoff", None), ("turn", "me"))),
        Problem(tuple(ActionDefinition(transition.action, (), (no_payoff,)) for transition in transitions)),
        TransitionModel(transitions),
        Players(("me",), "turn", ("payoff",)),
    )


def pay(payoff: float) -> PythonRule:
    return PythonRule(f"payoff = {payoff!r}")


def trust() -> Domain:
    """A plays safe (0.5 each) or risky; after risky, B plays punish (A 0, B 1) or reward (A 1, B 0)."""
    unset = PythonRule("payoff['A'] is None")

    def payoffs(a: float, b: float) -> PythonRule:
        return PythonRule(f"payoff['A'] = {a!r}\npayoff['B'] = {b!r}")

    return Domain(
        "trust",
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A"))),
        Problem(
            (
                ActionDefinition("safe", (), (unset, PythonRule("turn == 'A'"))),
                ActionDefinition("risky", (), (unset, PythonRule("turn == 'A'"))),
                ActionDefinition("punish", (), (unset, PythonRule("turn == 'B'"))),
                ActionDefinition("reward", (), (unset, PythonRule("turn == 'B'"))),
            )
        ),
        TransitionModel(
            (
                Transition("safe", (Branch(1.0, payoffs(0.5, 0.5)),)),
                Transition("risky", (Branch(1.0, PythonRule("turn = 'B'")),)),
                Transition("punish", (Branch(1.0, payoffs(0.0, 1.0)),)),
                Transition("reward", (Branch(1.0, payoffs(1.0, 0.0)),)),
            )
        ),
        Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
    )


def test_action_values_give_each_legal_action_its_value_for_the_player_to_act() -> None:
    domain = trust()
    after_risky = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))

    assert new_exact_search().action_values(domain, domain.initial_state) == (
        (Action("safe", ()), 0.5),
        (Action("risky", ()), 0.0),
    )
    assert new_exact_search().action_values(domain, after_risky) == (
        (Action("punish", ()), 1.0),
        (Action("reward", ()), 0.0),
    )


def test_value_gives_every_player_payoff_under_perfect_play() -> None:
    domain = trust()
    finished = State((("payoff(A)", 1.0), ("payoff(B)", 0.0), ("turn", "B")))

    assert new_exact_search().value(domain, domain.initial_state) == (0.5, 0.5)
    assert new_exact_search().value(domain, finished) == (1.0, 0.0)


def test_the_winning_action_is_optimal() -> None:
    domain = one_move(Transition("lose", (Branch(1.0, pay(0.0)),)), Transition("win", (Branch(1.0, pay(1.0)),)))

    assert new_exact_search().optimal_actions(domain, domain.initial_state) == (Action("win", ()),)


def test_each_player_maximizes_their_own_payoff() -> None:
    domain = trust()
    after_risky = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))

    assert new_exact_search().optimal_actions(domain, domain.initial_state) == (Action("safe", ()),)
    assert new_exact_search().optimal_actions(domain, after_risky) == (Action("punish", ()),)


def test_outcomes_count_by_their_probability() -> None:
    gamble = Transition("gamble", (Branch(0.5, pay(1.0)), Branch(0.5, pay(0.0))))
    better = one_move(gamble, Transition("safe", (Branch(1.0, pay(0.6)),)))
    equal = one_move(gamble, Transition("safe", (Branch(1.0, pay(0.5)),)))

    assert new_exact_search().optimal_actions(better, better.initial_state) == (Action("safe", ()),)
    assert new_exact_search().optimal_actions(equal, equal.initial_state) == (Action("gamble", ()), Action("safe", ()))


def test_positions_are_the_reachable_states_with_a_legal_action() -> None:
    domain = trust()

    assert new_exact_search().positions(domain) == (
        domain.initial_state,
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B"))),
    )


def test_a_copy_sent_to_another_process_leaves_its_values_behind() -> None:
    search, domain = new_exact_search(), one_move(Transition("win", (Branch(1.0, pay(1.0)),)))
    values = search.action_values(domain, domain.initial_state)

    copy = pickle.loads(pickle.dumps(search))

    assert copy.action_values(domain, domain.initial_state) == values


def test_a_state_without_legal_action_raises() -> None:
    domain = one_move(Transition("win", (Branch(1.0, pay(1.0)),)))

    with pytest.raises(ValueError, match="No legal action"):
        new_exact_search().optimal_actions(domain, State((("payoff", 1.0), ("turn", "me"))))
