import pytest

from openmind.agent.model.domain import Domain
from openmind.agent.service.timekeeper import Timekeeper
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.python_rule import PythonRule
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.players import Players
from openmind.world.model.state import State

#: A timeout rule: the player whose time ran out gets 0.0, the other 1.0.
TIMEOUT = PythonRule("payoff['A'] = 0.0 if flagged == 'A' else 1.0\npayoff['B'] = 0.0 if flagged == 'B' else 1.0")


def alternating_steps(timeout: PythonRule | None, stages: int = 4) -> Domain:
    """A and B take turns stepping; once the steps reach `stages`, the game is a draw."""
    return Domain(
        "alternating steps",
        State((("payoff(A)", None), ("payoff(B)", None), ("stage", 0), ("turn", "A"))),
        Problem((ActionDefinition("step", (), (PythonRule("payoff['A'] is None"),)),)),
        TransitionModel(
            (
                Transition(
                    "step",
                    (
                        Branch(
                            1.0,
                            PythonRule(
                                "stage = stage + 1\n"
                                "turn = 'B' if turn == 'A' else 'A'\n"
                                f"if stage >= {stages}:\n"
                                "    payoff['A'] = 0.5\n"
                                "    payoff['B'] = 0.5"
                            ),
                        ),
                    ),
                ),
            )
        ),
        Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
        timeout=timeout,
    )


def test_each_player_starts_with_the_time_control_s_clock() -> None:
    clocks = Timekeeper(create_rule_caller()).clocks(alternating_steps(TIMEOUT), TimeControl(60.0, 1.0))

    assert clocks == (Clock(60.0, 1.0), Clock(60.0, 1.0))


def test_a_domain_without_a_timeout_rule_can_t_be_played_on_a_clock() -> None:
    with pytest.raises(ValueError, match="no timeout rule"):
        Timekeeper(create_rule_caller()).clocks(alternating_steps(None), TimeControl(60.0))


def test_a_choice_is_timed() -> None:
    assert Timekeeper(create_rule_caller(), Ticking(0.5)).timed(lambda: "chosen") == ("chosen", 0.5)


def test_the_timeout_rule_says_what_running_out_of_time_does() -> None:
    domain = alternating_steps(TIMEOUT)

    state = Timekeeper(create_rule_caller()).flag(domain, domain.initial_state, "A")

    assert (dict(state.variables)["payoff(A)"], dict(state.variables)["payoff(B)"]) == (0.0, 1.0)
