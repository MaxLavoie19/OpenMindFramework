import logging

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain
from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.factory.rbs_factory import create_value_generator
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_settings import ValueSettings
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")

LINES = (
    *(((row, 1), (row, 2), (row, 3)) for row in (1, 2, 3)),
    *(((1, col), (2, col), (3, col)) for col in (1, 2, 3)),
    ((1, 1), (2, 2), (3, 3)),
    ((1, 3), (2, 2), (3, 1)),
)


def reachable(domain: Domain) -> list[State]:
    """Every position a game of tic-tac-toe can reach, in the order first reached."""
    solver, predictor = SolverBuilder().build(), PredictorBuilder().build()
    seen, frontier = {domain.initial_state: None}, [domain.initial_state]
    while frontier:
        following: list[State] = []
        for state in frontier:
            for action in solver.solve(domain.problem, state):
                for outcome, _ in predictor.predict(domain.transitions, state, action).outcomes:
                    if outcome not in seen:
                        seen[outcome] = None
                        following.append(outcome)
        frontier = following
    return list(seen)


def wins(cells: dict[tuple[int, int], object], player: str) -> bool:
    return any(all(cells[cell] == player for cell in line) for line in LINES)


def guaranteed_win(cells: dict[tuple[int, int], object]) -> bool:
    """Whether, O to act, X wins on its next move whatever O plays, worked out on the board alone."""
    empty = [cell for cell, mark in cells.items() if mark is None]
    if not empty or wins(cells, "O") or wins(cells, "X"):
        return False
    for reply in empty:
        after = {**cells, reply: "O"}
        if wins(after, "O") or not any(wins({**after, move: "X"}, "X") for move in empty if move != reply):
            return False
    return True


def test_the_search_deduces_a_guaranteed_win_by_looking_two_actions_ahead(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind")
    domain = create_tictactoe_domain()
    rows: list[PositionRow] = []
    for state in reachable(domain):
        variables = dict(state.variables)
        if variables["turn"] != "O" or variables["payoff(X)"] is not None:
            continue
        cells = {(row, col): variables[f"cell({row},{col})"] for row in (1, 2, 3) for col in (1, 2, 3)}
        rows.append(PositionRow(state, "X", 1.0 if guaranteed_win(cells) else 0.0))
    training, held_out = rows[::2], rows[1::2]
    settings = ValueSettings(
        prices=(0.1, 0.01, 0.001), max_steps=1000, tolerance=1e-6, seconds=600.0, memory_bytes=2 * 1024**3, candidates=20_000
    )

    result = create_value_generator().generate(domain, training, held_out, settings)

    no_rule, chosen = result.fits[0], result.chosen
    assert sum(row.target for row in rows) > 0
    assert chosen is not None and chosen.held_out_loss is not None and no_rule.held_out_loss is not None
    assert any("lambda v2:" in rule.term.source for rule in result.value_base.rules)
    assert chosen.held_out_loss < no_rule.held_out_loss / 2
