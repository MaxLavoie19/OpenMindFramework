from datetime import datetime

import pytest

from openmind.knowledge.constant.knowledge_constant import DECLARATION, INFERENCE
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT, POSITION
from openmind.knowledge.mapper.rule_record_json_mapper import RuleRecordJsonMapper
from openmind.knowledge.model.source import Source
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.state import State


def rook_moves_straight(state: State, **parameters: object) -> bool:
    """A rule a project wrote as a function of its own, at a module's top level where a worker can find it."""
    return state is not None and (parameters.get("rank") is None or parameters.get("file") is None)


def test_a_rule_written_as_source_goes_out_and_comes_back_with_everything_it_carries() -> None:
    mapper = RuleRecordJsonMapper()
    when = datetime(2026, 9, 17, 14, 30)
    rule = RuleRecord(
        "a cell is played only when it is empty",
        CONSTRAINT,
        PythonRule("cell[row, col] is None"),
        Source(DECLARATION, (("application", "the tic-tac-toe project"),), when),
        (("tictactoe", 1.0), ("four in a row", 0.75)),
        action="play",
        open=True,
        tags=(("topic", "legality"),),
    )

    back = mapper.from_line(mapper.to_line(rule))

    assert back == rule
    assert back.weight("four in a row") == 0.75
    assert back.weight("chess") == 0.0


def test_a_rule_the_project_gave_as_a_function_comes_back_as_that_very_function() -> None:
    mapper = RuleRecordJsonMapper()
    rule = RuleRecord("a rook moves straight", CONSTRAINT, rook_moves_straight, Source(DECLARATION), (("chess", 1.0),))

    back = mapper.from_line(mapper.to_line(rule))

    assert back.rule is rook_moves_straight


def test_a_function_that_is_no_longer_there_is_refused_rather_than_coming_back_missing() -> None:
    mapper = RuleRecordJsonMapper()
    data = mapper.to_data(RuleRecord("a rook moves straight", CONSTRAINT, rook_moves_straight, Source(DECLARATION)))
    data["function"] = "moved_away"

    with pytest.raises(ValueError, match="a rook moves straight"):
        mapper.from_data(data)


def test_a_heuristic_keeps_which_kind_it_is_and_where_it_came_from() -> None:
    mapper = RuleRecordJsonMapper()
    rule = RuleRecord(
        "a player with more moves is better placed",
        POSITION,
        PythonRule("len(here.moves(me))"),
        Source(INFERENCE, (("method", "deduction"), ("game", "0031"), ("ply", 12))),
        (("tictactoe", 0.4),),
    )

    back = mapper.from_line(mapper.to_line(rule))

    assert back.kind == POSITION
    assert back.source.mechanism == INFERENCE
    assert (back.source.parameter("game"), back.source.parameter("ply")) == ("0031", 12)
