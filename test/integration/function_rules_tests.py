"""A domain written the way a project writes one: its own Python functions, handed to OMF, instead of rule source.

The game is the strip of `rbs/service/consequence_library_tests.py`, rule for rule, so the two can be compared.
"""

import pickle

import pytest

from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.model.domain import Domain
from openmind.agent.service.game_recorder import GameRecorder
from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.state_domain import StateDomain
from openmind.csp.model.variable import Variable
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rbs.service.consequence_library_tests import place, position, strip_domain
from openmind.rule.factory.rule_factory import create_rule_caller
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


def function_strip() -> Domain:
    """The strip game with every rule written as one of this module's functions."""
    return (
        DomainBuilder()
        .with_name("strip")
        .with_initial_state(position({}, "X"))
        .with_problem(Problem((ActionDefinition("place", (Variable("col", StateDomain(free_columns)),), (is_empty,)),)))
        .with_transitions(TransitionModel((Transition("place", (Branch(1.0, place_mark),)),)))
        .with_players(Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)")))
        .with_ending(why_it_ended)
        .with_record(game_record)
        .build()
    )


def test_a_domain_written_as_functions_solves_and_plays_like_the_same_rules_as_source() -> None:
    written, source = function_strip(), strip_domain()
    solver, predictor = SolverBuilder().build(), PredictorBuilder().build()
    state = position({1: "X", 2: "O"}, "X")

    actions = solver.solve(written.problem, state)
    outcome = predictor.predict(written.transitions, state, place(3)).outcomes

    assert actions == solver.solve(source.problem, state)
    assert outcome == predictor.predict(source.transitions, state, place(3)).outcomes
    # A line ends the game in both, and nothing is legal afterwards.
    (won, _), = predictor.predict(written.transitions, position({1: "X"}, "X"), place(2)).outcomes
    assert won == predictor.predict(source.transitions, position({1: "X"}, "X"), place(2)).outcomes[0][0]
    assert not solver.solve(written.problem, won)


def test_the_ending_and_the_record_come_from_the_project_s_functions() -> None:
    recorder, domain = GameRecorder(create_rule_caller()), function_strip()
    won = place_mark(position({1: "X"}, "X"), 2)

    assert recorder.ending(domain, won) == "X made a line"
    assert recorder.ending(domain, position({1: "X"}, "O")) is None
    assert recorder.record(domain, (place(1), place(3))) == "1 3"


def test_a_function_no_worker_could_find_is_rejected() -> None:
    def inside() -> bool:
        return True

    with pytest.raises(ValueError, match="can't be found by a worker process"):
        DomainBuilder().with_ending(inside)
    with pytest.raises(ValueError, match="can't be found by a worker process"):
        DomainBuilder().with_record(lambda state, actions: "")


def test_a_domain_of_functions_travels_to_another_process() -> None:
    domain = function_strip()

    copy = pickle.loads(pickle.dumps(domain))

    solver = SolverBuilder().build()
    assert copy == domain
    assert solver.solve(copy.problem, position({1: "X"}, "O")) == solver.solve(domain.problem, position({1: "X"}, "O"))


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
