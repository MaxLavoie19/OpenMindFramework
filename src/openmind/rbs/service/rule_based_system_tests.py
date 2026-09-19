from collections.abc import Callable

import pytest

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction

type Game = Callable[[str], RuleBasedSystem]


def test_a_declared_game_says_where_it_starts_and_who_plays(game: Game) -> None:
    rbs = game("tictactoe")

    assert rbs.context == "tictactoe"
    assert rbs.start().value("turn") == "X"
    assert rbs.players().names == ("X", "O")
    assert rbs.players().to_act == "turn"


def test_the_legal_moves_are_solved_over_the_constraint_rules(game: Game) -> None:
    rbs = game("tictactoe")

    actions = rbs.actions(rbs.start())

    assert len(actions) == 9
    assert {action.name for action in actions} == {"place"}
    assert dict(actions[0].parameters) == {"row": 1, "col": 1}


def test_a_move_leads_where_its_effects_rule_says(game: Game) -> None:
    rbs = game("tictactoe")

    ((outcome, probability),) = rbs.outcomes(rbs.start(), Action("place", (("col", 2), ("row", 2)))).outcomes

    assert probability == 1.0
    assert outcome.model("cell")[2, 2] == "X"
    assert outcome.value("turn") == "O"


def test_a_game_is_played_to_its_end_through_the_rbs_alone(game: Game) -> None:
    rbs = game("tictactoe")
    state, played = rbs.start(), 0

    while actions := rbs.actions(state):
        state = rbs.outcomes(state, actions[0]).outcomes[0][0]
        played += 1

    assert played == 7
    assert state.model("payoff")["X"] == 1.0


def test_sudoku_s_all_different_constraints_and_computed_values_come_through(game: Game) -> None:
    rbs = game("sudoku")

    actions = rbs.actions(rbs.start(), limit=5)

    assert actions
    assert all(action.name == "fill" for action in actions)


def test_a_variant_is_its_own_context_with_its_own_rules(game: Game) -> None:
    rbs = game("tictactoe/fourinarow")

    assert rbs.context == "tictactoe/fourinarow"
    assert len(rbs.actions(rbs.start())) == 7


def test_the_games_of_one_knowledge_base_are_kept_apart(game: Game) -> None:
    tictactoe, sudoku = game("tictactoe"), game("sudoku")

    assert len(tictactoe.actions(tictactoe.start())) == 9
    assert {action.name for action in sudoku.actions(sudoku.start(), limit=1)} == {"fill"}


def test_where_players_act_at_once_their_choices_lead_somewhere_together(game: Game) -> None:
    rbs = game("prisonersdilemma/simultaneous")
    start = rbs.start()
    first, second = (rbs.actions(start, player=name)[0] for name in rbs.players().names)

    outcomes = rbs.joint_outcomes(start, JointAction((("A", first), ("B", second))))

    assert outcomes.outcomes


def test_each_player_to_act_at_once_gets_its_own_legal_moves(game: Game) -> None:
    rbs = game("rockpaperscissors")

    legal = rbs.joint_actions(rbs.start())

    assert [index for index, _ in legal] == [0, 1]
    assert all(len(actions) == 3 for _, actions in legal)


def test_a_game_that_says_what_running_out_of_time_does_can_be_played_on_a_clock(game: Game) -> None:
    rbs = game("tictactoe")

    assert rbs.timed()
    flagged = rbs.flagged(rbs.start(), flagged="X")

    assert flagged is not None
    assert flagged.model("payoff")["X"] == 0.0


def test_a_game_that_says_nothing_about_time_can_t_be_played_on_a_clock(game: Game) -> None:
    rbs = game("sudoku")

    assert rbs.timed() is False
    assert rbs.flagged(rbs.start(), flagged="solver") is None


def test_each_grid_s_empty_value_is_what_the_game_says_it_is(game: Game) -> None:
    rbs = game("tictactoe")

    assert rbs.empty("cell") is None
    assert rbs.empties() == (("cell", None),)


def test_a_context_the_knowledge_base_holds_no_rule_for_is_no_game(knowledge: KnowledgeBase) -> None:
    rbs = create_rule_based_system(knowledge, "nothing")

    with pytest.raises(ValueError, match="no rule saying where the game starts"):
        rbs.start()


def test_a_context_without_heuristics_judges_neither_position_nor_move(game: Game) -> None:
    rbs = game("tictactoe")

    assert rbs.value(rbs.start(), "X") is None
    assert rbs.values(rbs.start()) is None
    assert set(rbs.rate(rbs.start(), rbs.actions(rbs.start()))) == {None}


def test_the_search_s_statistics_are_summed_over_the_game_s_actions(declared: object) -> None:
    from openmind.rule.model.python_rule import PythonRule
    from openmind.world.model.state import State

    bit = PythonRule("(0, 1)")
    rbs = declared(  # type: ignore[operator]
        State.of(payoff=None, turn="me"),
        legal={"set": (PythonRule("a == 1"),), "put": (PythonRule("a == 1"),)},
        outcomes={"set": ((1.0, PythonRule("payoff = 1.0")),), "put": ((1.0, PythonRule("payoff = 0.0")),)},
        parameters={"set": {"a": bit, "b": bit}, "put": {"a": bit, "b": bit}},
    )

    actions, statistics = rbs.actions_with_statistics(rbs.start())

    assert len(actions) == 4
    assert (statistics.solutions, statistics.assignments) == (4, 4)
