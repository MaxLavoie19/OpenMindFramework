"""A game declared the way a project declares one: its own Python functions, handed to OMF, instead of rule source.

The game is the strip of `rbs/service/consequence_library_tests.py`, rule for rule, so the two can be compared.
"""

import pickle

import pytest

from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.consequence_library_tests import Declare, place, position, strip_domain
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value

COLUMNS = (1, 2, 3, 4)


def marks(state: State) -> dict[int, Value]:
    values = dict(state.variables)
    return {col: values[f"cell(1,{col})"] for col in COLUMNS}


def is_over(state: State) -> bool:
    return dict(state.variables)["payoff(X)"] is not None


def free_columns(state: State) -> tuple[Value, ...]:
    """The columns a mark can still go in: the state domain of the `col` parameter."""
    return () if is_over(state) else tuple(col for col, mark in marks(state).items() if mark is None)


def is_empty(state: State, col: int) -> bool:
    """The constraint: the cell is free and the game goes on."""
    return not is_over(state) and marks(state)[col] is None


def place_mark(state: State, col: int) -> State:
    """The effects: the player to act marks the column, two marks side by side win, a full strip draws."""
    values = dict(state.variables)
    turn, other = values["turn"], "O" if values["turn"] == "X" else "X"
    values[f"cell(1,{col})"] = turn
    placed = {column: values[f"cell(1,{column})"] for column in COLUMNS}
    if any(placed[column] == turn == placed[column + 1] for column in COLUMNS[:-1]):
        values[f"payoff({turn})"], values[f"payoff({other})"] = 1.0, 0.0
    elif all(mark is not None for mark in placed.values()):
        values["payoff(X)"] = values["payoff(O)"] = 0.5
    values["turn"] = other
    return State(tuple(sorted(values.items())))


def why_it_ended(state: State) -> str | None:
    """The ending: who made a line, a full strip, or nothing while the game goes on."""
    values = dict(state.variables)
    if values["payoff(X)"] is None:
        return None
    if values["payoff(X)"] == 0.5:
        return "a full strip"
    return f"{'X' if values['payoff(X)'] == 1.0 else 'O'} made a line"


def game_record(state: State, actions: tuple) -> str:  # type: ignore[type-arg]
    """The record: the columns played, in order."""
    return " ".join(str(dict(action.parameters)["col"]) for action in actions)


def function_strip(knowledge: KnowledgeBase) -> RuleBasedSystem:
    """The strip game with every rule declared as one of this module's functions."""
    declarer = RuleDeclarer(knowledge, "function strip", rule_caller=create_rule_caller())
    declarer.starts_at(position({}, "X"))
    declarer.played_by(Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)")))
    declarer.values("place", "col", free_columns)
    declarer.constraint("place", 1, is_empty)
    declarer.leads_to("place", place_mark)
    declarer.ending(why_it_ended)
    declarer.record(game_record)
    return create_rule_based_system(knowledge, declarer.done())


def test_a_game_declared_as_functions_solves_and_plays_like_the_same_rules_as_source(
    declared: Declare, knowledge: KnowledgeBase
) -> None:
    written, source = function_strip(knowledge), strip_domain(declared)
    state = position({1: "X", 2: "O"}, "X")

    actions = written.actions(state)
    outcome = written.outcomes(state, place(3)).outcomes

    assert actions == source.actions(state)
    assert outcome == source.outcomes(state, place(3)).outcomes
    # A line ends the game in both, and nothing is legal afterwards.
    ((won, _),) = written.outcomes(position({1: "X"}, "X"), place(2)).outcomes
    assert won == source.outcomes(position({1: "X"}, "X"), place(2)).outcomes[0][0]
    assert not written.actions(won)


def test_the_ending_and_the_record_come_from_the_project_s_functions(knowledge: KnowledgeBase) -> None:
    rbs = function_strip(knowledge)
    won = place_mark(position({1: "X"}, "X"), 2)

    assert rbs.ended(won) == "X made a line"
    assert rbs.ended(position({1: "X"}, "O")) is None
    assert rbs.record((place(1), place(3))) == "1 3"


def test_a_function_no_worker_could_find_is_rejected(knowledge: KnowledgeBase) -> None:
    def inside() -> bool:
        return True

    declarer = RuleDeclarer(knowledge, "strip", rule_caller=create_rule_caller())
    with pytest.raises(ValueError, match="can't be found by a worker process"):
        declarer.ending(inside)
    with pytest.raises(ValueError, match="can't be found by a worker process"):
        declarer.record(lambda state, actions: "")


def test_a_game_of_functions_travels_to_another_process(knowledge: KnowledgeBase) -> None:
    rbs = function_strip(knowledge)
    state = position({1: "X"}, "O")

    copy = pickle.loads(pickle.dumps(rbs))

    assert copy.rules == rbs.rules
    assert copy.actions(state) == rbs.actions(state)


def older_record(state: State, actions: tuple) -> str:  # type: ignore[type-arg]
    """A record rule written before the flag and the payoffs were offered."""
    return str(len(actions))


def keyword_record(state: State, **parameters: object) -> str:
    """A record rule taking whatever it's given."""
    return ",".join(sorted(parameters))


def test_a_function_gets_only_the_parameters_its_signature_names_or_all_of_them_when_it_takes_any_keyword() -> None:
    caller = create_rule_caller()
    state = State((("turn", "X"),))
    offered = {"actions": (), "flagged": None, "payoffs": None}

    assert caller.value(older_record, state, offered) == "0"
    assert caller.value(keyword_record, state, offered) == "actions,flagged,payoffs"
