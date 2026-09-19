import logging
from collections.abc import Callable

import pytest

from openmind.knowledge.constant.knowledge_constant import COPY
from openmind.knowledge.constant.rule_kind_constant import CONSTRAINT
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.testing.plugin.game_fixtures import simulation_rules
from openmind.world.model.action import Action

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedGame]

MY_TURN = "tictactoe without place is legal, 1"
EMPTY_CELL = "tictactoe without place is legal, 4"


def test_the_relaxations_are_each_constraint_dropped(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    assert GameRelaxer(knowledge).relaxations("tictactoe") == (
        MY_TURN,
        "tictactoe without place is legal, 2",
        "tictactoe without place is legal, 3",
        EMPTY_CELL,
    )


def test_a_relaxation_is_a_game_of_its_own_with_one_constraint_fewer(game: Game, knowledge: KnowledgeBase) -> None:
    rbs = game("tictactoe")
    marked = rbs.outcomes(rbs.start(), Action("place", (("col", 1), ("row", 1)))).outcomes[0][0]

    relaxed = create_rule_based_game(knowledge, GameRelaxer(knowledge).relax("tictactoe", EMPTY_CELL))

    # Without "the cell is empty", the marked cell can be played again.
    assert len(rbs.actions(marked)) == 8
    assert len(relaxed.actions(marked)) == 9
    assert len(relaxed.rules) == len(rbs.rules) - 1


def test_dropping_the_turn_constraint_lets_the_other_player_act_now(game: Game, knowledge: KnowledgeBase) -> None:
    rbs = game("tictactoe")

    relaxed = create_rule_based_game(knowledge, GameRelaxer(knowledge).relax("tictactoe", MY_TURN))

    assert rbs.acting(rbs.start()) == (0,)
    assert relaxed.acting(relaxed.start()) == (0, 1)


def test_relaxing_leaves_the_game_it_relaxes_as_it_was(game: Game, knowledge: KnowledgeBase) -> None:
    rbs = game("tictactoe")
    before = simulation_rules(knowledge, "tictactoe", (CONSTRAINT,))

    GameRelaxer(knowledge).relax("tictactoe", EMPTY_CELL)

    assert simulation_rules(knowledge, "tictactoe", (CONSTRAINT,)) == before
    assert len(create_rule_based_game(knowledge, "tictactoe").actions(rbs.start())) == 9


def test_a_relaxation_s_ruleset_is_an_open_copy_of_the_game_s(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    GameRelaxer(knowledge).relax("tictactoe", EMPTY_CELL)

    (relaxed,) = knowledge.rulesets(knowledge.context_named(EMPTY_CELL).id)  # type: ignore[union-attr]
    assert not knowledge.frozen_ruleset(relaxed)
    assert relaxed.source.mechanism == knowledge.mechanism_named(COPY).id  # type: ignore[union-attr]


def test_a_name_that_is_not_one_of_the_game_s_relaxations_is_refused(game: Game, knowledge: KnowledgeBase) -> None:
    game("tictactoe")

    with pytest.raises(ValueError, match="isn't a relaxation of tictactoe"):
        GameRelaxer(knowledge).relax("tictactoe", "tictactoe without gravity")


def test_relaxing_is_logged_with_the_rules_the_relaxation_holds(
    game: Game, knowledge: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    rbs = game("tictactoe")

    GameRelaxer(knowledge).relax("tictactoe", EMPTY_CELL)

    assert f"Relaxed tictactoe into {EMPTY_CELL}: {len(rbs.rules) - 1} rules" in caplog.messages
