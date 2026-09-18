import logging
from collections.abc import Callable

import pytest

from openmind.doxastic.constant.doxastic_constant import ASSUMED
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

EMPTY_CELL = "tictactoe without place is legal, 3"
PASSING = "tictactoe where a player may pass"


def relaxer(knowledge: KnowledgeBase) -> GameRelaxer:
    return GameRelaxer(knowledge, create_rule_caller())


def test_the_relaxations_are_each_constraint_dropped_and_a_player_passing(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    assert relaxer(knowledge).relaxations("tictactoe") == (
        "tictactoe without place is legal, 1",
        "tictactoe without place is legal, 2",
        EMPTY_CELL,
        PASSING,
    )


def test_a_relaxation_is_a_game_of_its_own_with_one_constraint_fewer(game: Game, knowledge: KnowledgeBase) -> None:
    rbs = game("tictactoe")
    marked = rbs.outcomes(rbs.start(), Action("place", (("col", 1), ("row", 1)))).outcomes[0][0]

    relaxed = create_rule_based_system(knowledge, relaxer(knowledge).relax("tictactoe", EMPTY_CELL))

    # Without "the cell is empty", the marked cell can be played again.
    assert len(rbs.actions(marked)) == 8
    assert len(relaxed.actions(marked)) == 9
    assert len(relaxed.rules) == len(rbs.rules) - 1


def test_relaxing_leaves_the_game_it_relaxes_as_it_was(game: Game, knowledge: KnowledgeBase) -> None:
    rbs = game("tictactoe")
    before = len(knowledge.rules("tictactoe", (CONSTRAINT,)))

    relaxer(knowledge).relax("tictactoe", EMPTY_CELL)

    assert len(knowledge.rules("tictactoe", (CONSTRAINT,))) == before
    assert len(create_rule_based_system(knowledge, "tictactoe").actions(rbs.start())) == 9


def test_in_the_passing_relaxation_a_player_may_hand_the_turn_over(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    relaxed = create_rule_based_system(knowledge, relaxer(knowledge).relax("tictactoe", PASSING))
    start = relaxed.start()
    ((after, _),) = relaxed.outcomes(start, Action("pass", ())).outcomes

    assert Action("pass", ()) in relaxed.actions(start)
    assert (dict(start.variables)["turn"], dict(after.variables)["turn"]) == ("X", "O")
    assert after.variables != start.variables


def test_the_rules_a_relaxation_invents_are_assumed_and_live_in_it_alone(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    relaxer(knowledge).relax("tictactoe", PASSING)

    invented = [rule for rule in knowledge.rules(PASSING) if rule.action == "pass"]
    assert invented and all(rule.provenance.source == ASSUMED for rule in invented)
    assert all(rule.contexts == ((PASSING, rule.weight(PASSING)),) for rule in invented)
    assert not [rule for rule in knowledge.rules("tictactoe") if rule.action == "pass"]


def test_a_name_that_is_not_one_of_the_game_s_relaxations_is_refused(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    with pytest.raises(ValueError, match="isn't a relaxation of tictactoe"):
        relaxer(knowledge).relax("tictactoe", "tictactoe without gravity")


def test_relaxing_is_logged_with_the_rules_the_relaxation_holds(
    game: Game, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    rbs = game("tictactoe")

    relaxer(knowledge).relax("tictactoe", EMPTY_CELL)

    assert f"Relaxed tictactoe into {EMPTY_CELL}: {len(rbs.rules) - 1} rules" in caplog.messages
