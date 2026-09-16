import logging
from dataclasses import replace

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.inference.model.relaxation import Relaxation
from openmind.inference.service.domain_relaxer import DomainRelaxer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action
from openmind.world.model.players import Players

pytestmark = pytest.mark.log_level("INFO")


def new_relaxer() -> DomainRelaxer:
    return DomainRelaxer(create_rule_caller())


def columns(state: object) -> dict[str, object]:
    return dict(state.variables)  # type: ignore[attr-defined]


def test_the_rules_offer_a_relaxation_per_constraint_and_a_pass() -> None:
    relaxations = new_relaxer().relaxations(strip_domain())

    names = [relaxation.name for relaxation in relaxations]
    assert names == [
        "place without payoff['X'] is None",
        "place without payoff['O'] is None",
        "place without cell[1, col] is None",
        "a player may pass",
    ]
    assert [relaxation.dropped for relaxation in relaxations[:3]] == [(("place", 0),), (("place", 1),), (("place", 2),)]


def test_dropping_a_constraint_lets_the_solver_offer_what_it_forbade(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.inference")
    domain, solver = strip_domain(), create_solver()
    marked = position({1: "X", 2: "O"}, "X")

    relaxed = new_relaxer().relax(domain, Relaxation("marks can be overwritten", dropped=(("place", 2),)))

    assert [action.parameters for action in solver.solve(relaxed.problem, marked)] == [
        (("col", 1),),
        (("col", 2),),
        (("col", 3),),
        (("col", 4),),
    ]
    assert len(solver.solve(domain.problem, marked)) == 2
    assert "Relaxed strip: marks can be overwritten" in caplog.messages


def test_passing_hands_the_turn_over_and_ends_with_the_game() -> None:
    domain, solver, predictor = strip_domain(), create_solver(), create_predictor()

    relaxed = new_relaxer().relax(domain, Relaxation("a player may pass", passing=True))

    passes = [action for action in solver.solve(relaxed.problem, position({1: "X"}, "O")) if action.name == "pass"]
    ((after, _),) = predictor.predict(relaxed.transitions, position({1: "X"}, "O"), Action("pass", ())).outcomes
    assert passes == [Action("pass", ())]
    assert columns(after)["turn"] == "X" and columns(after)["cell(1,1)"] == "X"
    # The game being over, no move and no pass are left.
    over = position({1: "X", 2: "X"}, "O")
    assert solver.solve(relaxed.problem, replace(over, variables=tuple(sorted({**columns(over), "payoff(X)": 1.0, "payoff(O)": 0.0}.items())))) == ()


def test_widening_a_parameter_gives_it_the_values_the_rule_gives() -> None:
    domain = create_tictactoe_domain()
    anywhere = PythonRule("[1, 2, 3]")

    relaxed = new_relaxer().relax(domain, Relaxation("any row", widened=(("place", "row", anywhere),)))

    rows = {dict(action.parameters)["row"] for action in create_solver().solve(relaxed.problem, domain.initial_state)}
    assert rows == {1, 2, 3}
    assert relaxed.problem.actions[0].variables != domain.problem.actions[0].variables


def test_a_relaxed_copy_leaves_the_domain_as_it_is() -> None:
    domain = strip_domain()

    relaxed = new_relaxer().relax(domain, Relaxation("no legality", dropped=(("place", 2),), passing=True))

    assert domain == strip_domain() and relaxed != domain
    assert len(domain.problem.actions[0].constraints) == 3 and len(relaxed.problem.actions[0].constraints) == 2


def test_passing_needs_a_variable_naming_the_player_to_act() -> None:
    domain = replace(strip_domain(), players=Players(("X", "O"), "whose_turn", ("payoff(X)", "payoff(O)")))

    with pytest.raises(ValueError, match="no variable naming the player to act"):
        new_relaxer().relax(domain, Relaxation("a player may pass", passing=True))
