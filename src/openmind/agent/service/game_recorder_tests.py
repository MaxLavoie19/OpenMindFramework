import logging
from dataclasses import replace

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.service.game_recorder import GameRecorder
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action

pytestmark = pytest.mark.log_level("INFO")

ENDING = PythonRule("'a full board' if all(mark is not None for mark in cell.values()) else 'a line'")
RECORD = PythonRule("' '.join(f\"{dict(action.parameters)['row']}{dict(action.parameters)['col']}\" for action in actions)")


def new_recorder() -> GameRecorder:
    return GameRecorder(create_rule_caller())


def place(row: int, col: int) -> Action:
    return Action("place", (("col", col), ("row", row)))


def test_the_domain_s_rules_say_why_a_game_ended_and_record_its_actions() -> None:
    domain = replace(create_tictactoe_domain(), ending=ENDING, record=RECORD)

    assert new_recorder().ending(domain, domain.initial_state) == "a line"
    assert new_recorder().record(domain, (place(1, 1), place(2, 2))) == "11 22"


def test_a_domain_without_the_rules_gives_nothing() -> None:
    domain = create_tictactoe_domain()

    assert (new_recorder().ending(domain, domain.initial_state), new_recorder().record(domain, (place(1, 1),))) == (None, None)


def test_a_rule_that_raises_gives_nothing_and_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    domain = replace(create_tictactoe_domain(), ending=PythonRule("nothing[0]"), record=PythonRule("1 / 0"))

    assert (new_recorder().ending(domain, domain.initial_state), new_recorder().record(domain, ())) == (None, None)
    assert any("ending rule raised" in message for message in caplog.messages)
    assert any("record rule raised" in message for message in caplog.messages)
