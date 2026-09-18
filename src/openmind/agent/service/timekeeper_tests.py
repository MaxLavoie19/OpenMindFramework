from pathlib import Path

import pytest

from openmind.agent.service.timekeeper import Timekeeper
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.players import Players
from openmind.world.model.state import State

#: A timeout rule: the player whose time ran out gets 0.0, the other 1.0.
TIMEOUT = PythonRule("payoff['A'] = 0.0 if flagged == 'A' else 1.0\npayoff['B'] = 0.0 if flagged == 'B' else 1.0")


def alternating_steps(
    timeout: PythonRule | None, tmp_path: Path, stages: int = 4, context: str = "alternating steps"
) -> RuleBasedSystem:
    """A and B take turns stepping; once the steps reach `stages`, the game is a draw."""
    knowledge_base = create_knowledge_base(context, tmp_path)
    declarer = RuleDeclarer(knowledge_base, context)
    declarer.starts_at(State((("payoff(A)", None), ("payoff(B)", None), ("stage", 0), ("turn", "A"))))
    declarer.played_by(Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")))
    declarer.constraint("step", 1, PythonRule("payoff['A'] is None"))
    declarer.leads_to(
        "step",
        PythonRule(
            "stage = stage + 1\n"
            "turn = 'B' if turn == 'A' else 'A'\n"
            f"if stage >= {stages}:\n"
            "    payoff['A'] = 0.5\n"
            "    payoff['B'] = 0.5"
        ),
    )
    if timeout is not None:
        declarer.timeout(timeout)
    return create_rule_based_system(knowledge_base, declarer.done())


def test_each_player_starts_with_the_time_control_s_clock(tmp_path: Path) -> None:
    clocks = Timekeeper(create_rule_caller()).clocks(alternating_steps(TIMEOUT, tmp_path), TimeControl(60.0, 1.0))

    assert clocks == (Clock(60.0, 1.0), Clock(60.0, 1.0))


def test_a_domain_without_a_timeout_rule_can_t_be_played_on_a_clock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no timeout rule"):
        Timekeeper(create_rule_caller()).clocks(alternating_steps(None, tmp_path), TimeControl(60.0))


def test_a_choice_is_timed(tmp_path: Path) -> None:
    assert Timekeeper(create_rule_caller(), Ticking(0.5)).timed(lambda: "chosen") == ("chosen", 0.5)


def test_the_timeout_rule_says_what_running_out_of_time_does(tmp_path: Path) -> None:
    rbs = alternating_steps(TIMEOUT, tmp_path)

    state = Timekeeper(create_rule_caller()).flag(rbs, rbs.start(), "A")

    assert (dict(state.variables)["payoff(A)"], dict(state.variables)["payoff(B)"]) == (0.0, 1.0)
