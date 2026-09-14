import logging
from dataclasses import replace

import pytest

from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.service.advantage_contrast import AdvantageContrast
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.consequence_library_tests import place, position, strip_domain
from openmind.rbs.service.hypothesis_discoverer import HypothesisDiscoverer
from openmind.rbs.service.primitive_generator_tests import SETTINGS
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

WINS = PythonRule("win_chance(action) >= 1")

#: X to play, each with a winning move and a losing one, then two where every move wins.
POSITIONS = (
    {1: "X"}, {4: "X"}, {1: "X", 4: "O"}, {4: "X", 1: "O"}, {2: "X"}, {3: "X"},
    {1: "X", 3: "O"}, {4: "X", 2: "O"}, {2: "X", 1: "O"}, {3: "X", 4: "O"},
    {2: "X", 4: "O"}, {3: "X", 1: "O"},
)  # fmt: skip


def samples(positions: tuple[dict[int, str], ...] = POSITIONS) -> list[ActionSample]:
    """Every legal move, paying 1.0 when it makes two X side by side and 0.4 otherwise."""
    found = []
    for marks in positions:
        for col in (col for col in range(1, 5) if col not in marks):
            after = {**marks, col: "X"}
            wins = any(after.get(cell) == "X" and after.get(cell + 1) == "X" for cell in range(1, 4))
            found.append(ActionSample(position(marks, "X"), 0, place(col), 10, 1.0 if wins else 0.4))
    return found


def rows() -> tuple[ActionRow, ...]:
    return ActionRowMapper().to_rows(samples(), min_visits=1)


def new_discoverer() -> HypothesisDiscoverer:
    evaluator = ConditionEvaluator(
        RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build()
    )
    return HypothesisDiscoverer(evaluator, ActionRowMapper(), AdvantageContrast())


def test_conditions_separating_actions_within_states_become_hypotheses_best_score_first() -> None:
    settings = replace(SETTINGS, max_conditions=1)

    hypotheses = new_discoverer().discover(strip_domain(), "place", rows(), (PythonRule("col == 4"), WINS), (), settings)

    assert [(hypothesis.conditions, hypothesis.parent, hypothesis.direction, hypothesis.states) for hypothesis in hypotheses] == [
        ((WINS,), (), 1, 10),
        ((PythonRule("col == 4"),), (), -1, 6),
    ]
    assert hypotheses[0].effect == pytest.approx(0.6)
    assert hypotheses[0].expected_value == pytest.approx(1.0)


def test_a_condition_the_same_for_every_action_of_a_state_gates_others() -> None:
    first_cell_empty = PythonRule("cell[1, 1] == None")

    hypotheses = new_discoverer().discover(strip_domain(), "place", rows(), (first_cell_empty, WINS), (), SETTINGS)

    assert [(hypothesis.conditions, hypothesis.parent, hypothesis.states) for hypothesis in hypotheses] == [
        ((WINS,), (), 10),
        ((first_cell_empty, WINS), (first_cell_empty,), 5),
    ]


def test_a_goal_pattern_is_tried_whole() -> None:
    pattern = GoalPattern("place", (PythonRule("near(action, 0, -1) == me"),), 4)

    hypotheses = new_discoverer().discover(strip_domain(), "place", rows(), (), (pattern,), SETTINGS)

    assert [(hypothesis.conditions, hypothesis.direction) for hypothesis in hypotheses] == [(pattern.conditions, 1)]


def test_a_combination_one_of_its_conditions_adds_nothing_to_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    wins_or_last = PythonRule("win_chance(action) >= 1 or col == 4")

    hypotheses = new_discoverer().discover(strip_domain(), "place", rows(), (wins_or_last, WINS), (), SETTINGS)

    assert [hypothesis.conditions for hypothesis in hypotheses] == [(WINS,), (wins_or_last,)]
    assert (
        "place: skipped 1 combinations a condition adds nothing to, shortened 0 goal patterns, merged 0 hypotheses "
        "matching the same rows"
    ) in caplog.messages


def test_a_gate_that_adds_nothing_to_a_condition_is_skipped() -> None:
    can_win = PythonRule("wins(me) >= 1")
    without_a_win = ({1: "X", 2: "O"}, {4: "X", 3: "O"}, {2: "O"})
    with_losses = ActionRowMapper().to_rows(samples((*POSITIONS, *without_a_win)), min_visits=1)

    hypotheses = new_discoverer().discover(strip_domain(), "place", with_losses, (can_win, WINS), (), SETTINGS)

    assert [hypothesis.conditions for hypothesis in hypotheses] == [(WINS,)]


def test_a_goal_pattern_loses_the_conditions_that_leave_its_rows_the_same(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    left_is_mine = PythonRule("near(action, 0, -1) == me")
    pattern = GoalPattern("place", (PythonRule("turn == 'X'"), left_is_mine), 4)

    hypotheses = new_discoverer().discover(strip_domain(), "place", rows(), (), (pattern,), SETTINGS)

    assert [hypothesis.conditions for hypothesis in hypotheses] == [(left_is_mine,)]
    assert (
        "place: skipped 0 combinations a condition adds nothing to, shortened 1 goal patterns, merged 0 hypotheses "
        "matching the same rows"
    ) in caplog.messages


def test_hypotheses_matching_the_same_rows_are_merged_into_the_one_with_the_fewest_conditions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    # Neither condition alone matches only the winning moves; together they do.
    pattern = GoalPattern(
        "place", (PythonRule("win_chance(action) >= 1 or col == 4"), PythonRule("win_chance(action) >= 1 or col == 1")), 4
    )
    settings = replace(SETTINGS, max_conditions=1)

    hypotheses = new_discoverer().discover(strip_domain(), "place", rows(), (WINS,), (pattern,), settings)

    assert [hypothesis.conditions for hypothesis in hypotheses] == [(WINS,)]
    assert (
        "place: skipped 0 combinations a condition adds nothing to, shortened 0 goal patterns, merged 1 hypotheses "
        "matching the same rows"
    ) in caplog.messages


def test_a_payoff_near_the_best_seen_marks_a_priority_hypothesis() -> None:
    settings = replace(SETTINGS, max_conditions=1, confidence=0.5)

    (hypothesis,) = new_discoverer().discover(strip_domain(), "place", rows(), (WINS,), (), settings)

    assert hypothesis.priority is True
