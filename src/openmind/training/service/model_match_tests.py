from collections.abc import Callable

from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_model_registry
from openmind.model.service.model_registry import ModelRegistry
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.training.factory.training_factory import create_model_match
from openmind.training.model.match import Match
from openmind.training.model.self_play_settings import SelfPlaySettings

type Game = Callable[[str], RuleBasedGame]
type Linked = Callable[..., RuleRecord]

#: A few games played, not a budget honoured. A reading of a tic-tac-toe heuristic costs microseconds, so a
#: tenth of a second buys tens of thousands of nodes and a handful of games takes minutes.
PLAY = SelfPlaySettings(games=4, seconds=0.003, seed=1)


def test_a_model_never_played_has_no_accuracy_and_one_that_played_does(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """The whole point of the link: games were kept and the one belief anything selects on never heard of
    them, so a model could win everything and `best` would go on preferring whichever was registered first.

    Measured on the model and not on its mechanism. Two rulesets fitted the same way share a mechanism, so
    both sides of a match would land on one belief and their points would sum to the games played."""
    played = game("tictactoe")
    scorer = AccuracyScorer()
    one = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    other = _registered(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")

    assert scorer.accuracy(knowledge, one.id, one.context) is None, "never measured is not a low score"

    create_model_match().play(knowledge, played, one, other, PLAY)

    assert scorer.accuracy(knowledge, one.id, one.context) is not None
    assert scorer.accuracy(knowledge, other.id, other.context) is not None


def test_the_points_of_a_match_come_to_the_games_played(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """A win for one, a draw for a half, a loss for none — so between two models the points on offer are
    exactly one per game and nothing is created or lost in the scoring."""
    played = game("tictactoe")
    one = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    other = _registered(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")

    match = create_model_match().play(knowledge, played, one, other, PLAY)

    assert match is not None
    assert match.points[0] + match.points[1] == match.games
    assert sum(match.sides) == match.games, "every game was played from one seating or the other"
    assert abs(match.sides[0] - match.sides[1]) <= 1, "the seatings are split as evenly as the count allows"


def test_a_match_of_models_answering_different_questions_is_no_match(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """Two models are comparable by playing only where they answer the same question."""
    played = game("tictactoe")
    one = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    other = _registered(knowledge, heuristic, "a mover", "0.0", task=MOVE_VALUE)

    assert create_model_match().play(knowledge, played, one, other, PLAY) is None


def test_a_match_needs_a_game_of_two(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """A match of two is what this is, and a game of any other number is declined rather than guessed at.

    Sudoku has one player, so there is no second seat to put the other model in and no result that would
    mean anything. Declining says so; playing anyway would produce a number about nothing."""
    game("tictactoe")  # the heuristics are declared in its context
    alone = game("sudoku")
    one = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    other = _registered(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")

    assert len(alone.players().names) != 2, "the game this leans on is not one of two"
    assert create_model_match().play(knowledge, alone, one, other, PLAY) is None


def test_a_level_match_names_no_winner():
    """Equal points over any number of games is a match that chose nothing, which is worth saying rather than
    rounding into a winner. Built rather than played, because what is pinned here is the reading of a result
    and not whether two particular heuristics can draw."""
    level = Match("tictactoe", POSITION_VALUE, "one", "other", 4, (2.0, 2.0), (2, 2))
    won = Match("tictactoe", POSITION_VALUE, "one", "other", 4, (2.5, 1.5), (2, 2))

    assert level.decisive is False
    assert level.winner == "", "level is not a win for whoever was named first"
    assert won.decisive is True
    assert won.winner == "one"
    assert Match("tictactoe", POSITION_VALUE, "one", "other", 4, (1.5, 2.5), (2, 2)).winner == "other"


def test_what_a_match_measures_is_what_the_registry_selects_on(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """Games are the evidence and the registry is the selector, and after a match the second reads the
    first: `best` prefers whichever the games preferred."""
    played = game("tictactoe")
    registry = ModelRegistry(AccuracyScorer())
    one = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    other = _registered(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")

    match = create_model_match().play(knowledge, played, one, other, PLAY)

    assert match is not None
    chosen = registry.best(knowledge, played.context_id, POSITION_VALUE)
    assert chosen is not None
    if match.decisive:
        assert chosen.name == match.winner, "the model the games preferred is the one best gives back"


def _registered(
    knowledge: KnowledgeBase,
    heuristic: Linked,
    name: str,
    rule: str,
    task: str = POSITION_VALUE,
) -> ModelRecord:
    """A heuristic of its own, in a ruleset of its own, registered as a model of that task.

    A ruleset behind it and not a bare record: the outfitter loads a model by going to its ruleset, so a
    record naming a ruleset nobody declared is a model that will not play."""
    heuristic("tictactoe", name, PythonRule(rule), 1.0, task=task, ruleset_name=name)
    context = knowledge.context_named("tictactoe")
    ruleset = knowledge.ruleset_named(context.id, name)  # type: ignore[union-attr]
    return create_model_registry().register_ruleset(knowledge, ruleset, name)  # type: ignore[arg-type]


def test_every_registered_model_of_a_task_plays_every_other(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """What closes the loop: a model registered and never played is only a claim, and best prefers whichever
    was registered first among models nothing has measured."""
    played = game("tictactoe")
    _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    _registered(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")
    _registered(knowledge, heuristic, "counting marks", "sum(1 for at in cell if cell[at] == me)")

    matches = create_model_match().among(knowledge, played, POSITION_VALUE, PLAY)

    assert len(matches) == 3, "three models make three pairs"
    assert {one.task for one in matches} == {POSITION_VALUE}
    assert all(one.games > 0 for one in matches)


def test_a_model_with_nobody_to_play_is_not_measured_by_playing(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """Saying nothing about it is right: it has not been beaten and it has not won."""
    played = game("tictactoe")
    alone = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")

    assert create_model_match().among(knowledge, played, POSITION_VALUE, PLAY) == ()
    assert AccuracyScorer().accuracy(knowledge, alone.id, alone.context) is None


def test_playing_them_all_is_what_lets_the_registry_choose(game: Game, knowledge: KnowledgeBase, heuristic: Linked):
    """Before the matches every model is unmeasured and best falls back on the order they were registered.
    After them it answers from the games."""
    played = game("tictactoe")
    registry = ModelRegistry(AccuracyScorer())
    first = _registered(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    _registered(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")

    before = registry.best(knowledge, played.context_id, POSITION_VALUE)
    assert before is not None and before.id == first.id, "unmeasured, so the first registered"

    matches = create_model_match().among(knowledge, played, POSITION_VALUE, PLAY)
    after = registry.best(knowledge, played.context_id, POSITION_VALUE)

    assert after is not None
    assert AccuracyScorer().accuracy(knowledge, after.id, after.context) is not None, "chosen on games now"
    if matches and matches[0].decisive:
        assert after.name == matches[0].winner
