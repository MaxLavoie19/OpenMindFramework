import pytest

from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.csp.factory.csp_factory import create_solver
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


def new_exact_search() -> ExactSearch:
    names = VariableNameMapper()
    interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
    return ExactSearch(
        create_solver(),
        Predictor(interpreter, names, expression_text, action_text),
        StateReader(),
    )


def one_move(*transitions: Transition) -> Domain:
    """One player; every action is legal until the payoff is set, and every action sets it."""
    no_payoff = Equals(StateVariable("payoff"), Constant(None))
    return Domain(
        "one move",
        State((("payoff", None), ("turn", "me"))),
        Problem(tuple(ActionDefinition(transition.action, (), (no_payoff,)) for transition in transitions)),
        TransitionModel(transitions),
        Players(("me",), "turn", ("payoff",)),
    )


def pay(payoff: float) -> Assign:
    return Assign(StateVariable("payoff"), Constant(payoff))


def trust() -> Domain:
    """A plays safe (0.5 each) or risky; after risky, B plays punish (A 0, B 1) or reward (A 1, B 0)."""
    unset = Equals(StateVariable("payoff", (Constant("A"),)), Constant(None))
    a_to_act = Equals(StateVariable("turn"), Constant("A"))
    b_to_act = Equals(StateVariable("turn"), Constant("B"))

    def payoffs(a: float, b: float) -> tuple[Assign, Assign]:
        return (
            Assign(StateVariable("payoff", (Constant("A"),)), Constant(a)),
            Assign(StateVariable("payoff", (Constant("B"),)), Constant(b)),
        )

    return Domain(
        "trust",
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A"))),
        Problem(
            (
                ActionDefinition("safe", (), (unset, a_to_act)),
                ActionDefinition("risky", (), (unset, a_to_act)),
                ActionDefinition("punish", (), (unset, b_to_act)),
                ActionDefinition("reward", (), (unset, b_to_act)),
            )
        ),
        TransitionModel(
            (
                Transition("safe", (Branch(1.0, payoffs(0.5, 0.5)),)),
                Transition("risky", (Branch(1.0, (Assign(StateVariable("turn"), Constant("B")),)),)),
                Transition("punish", (Branch(1.0, payoffs(0.0, 1.0)),)),
                Transition("reward", (Branch(1.0, payoffs(1.0, 0.0)),)),
            )
        ),
        Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
    )


def test_the_winning_action_is_optimal() -> None:
    domain = one_move(Transition("lose", (Branch(1.0, (pay(0.0),)),)), Transition("win", (Branch(1.0, (pay(1.0),)),)))

    assert new_exact_search().optimal_actions(domain, domain.initial_state) == (Action("win", ()),)


def test_each_player_maximizes_their_own_payoff() -> None:
    domain = trust()
    after_risky = State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))

    assert new_exact_search().optimal_actions(domain, domain.initial_state) == (Action("safe", ()),)
    assert new_exact_search().optimal_actions(domain, after_risky) == (Action("punish", ()),)


def test_outcomes_count_by_their_probability() -> None:
    gamble = Transition("gamble", (Branch(0.5, (pay(1.0),)), Branch(0.5, (pay(0.0),))))
    better = one_move(gamble, Transition("safe", (Branch(1.0, (pay(0.6),)),)))
    equal = one_move(gamble, Transition("safe", (Branch(1.0, (pay(0.5),)),)))

    assert new_exact_search().optimal_actions(better, better.initial_state) == (Action("safe", ()),)
    assert new_exact_search().optimal_actions(equal, equal.initial_state) == (Action("gamble", ()), Action("safe", ()))


def test_positions_are_the_reachable_states_with_a_legal_action() -> None:
    domain = trust()

    assert new_exact_search().positions(domain) == (
        domain.initial_state,
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B"))),
    )


def test_a_state_without_legal_action_raises() -> None:
    domain = one_move(Transition("win", (Branch(1.0, (pay(1.0),)),)))

    with pytest.raises(ValueError, match="No legal action"):
        new_exact_search().optimal_actions(domain, State((("payoff", 1.0), ("turn", "me"))))
