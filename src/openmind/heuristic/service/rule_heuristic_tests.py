from collections.abc import Callable

from openmind.knowledge.constant.knowledge_constant import INFERENCE
from openmind.knowledge.constant.rule_kind_constant import MOVE, POSITION
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game, create_rule_based_system, create_rule_heuristic
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]


def test_a_position_is_worth_its_rules_readings_times_their_weights_summed(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "a mark on the middle cell", PythonRule("cell[2, 2] == me"), 0.75)
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.25)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    reader = create_rule_heuristic()
    game_played = create_rule_based_game(knowledge, "tictactoe")
    start = game_played.node(game_played.start())

    assert reader.value(rbs, start, "X") == 0.25
    assert reader.values(rbs, start, ("X", "O")) == (0.25, 0.25)
    assert [rule.name for rule, _ in reader.explain(rbs, start, "X")] == ["a mark on the middle cell", "a constant"]


def test_a_rule_that_cant_be_read_here_is_left_out_and_one_reading_nothing_adds_nothing(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "reads a model the game has not", PythonRule("lamp"), 1.0)
    heuristic("tictactoe", "reads nothing here", PythonRule("None"), 1.0)
    heuristic("tictactoe", "a constant", PythonRule("2.0"), 0.5)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    start = played.node(played.start())

    added = dict(create_rule_heuristic().explain(rbs, start, "X"))

    assert [rule.name for rule in added] == ["reads nothing here", "a constant"]
    assert create_rule_heuristic().value(rbs, start, "X") == 1.0


def test_a_move_is_rated_for_the_player_taking_it_and_its_parameters_are_read(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "the middle cell is worth taking", PythonRule("row == 2 and col == 2"), 1.0, MOVE)
    rbs = create_rule_based_system(knowledge, "tictactoe", MOVE_VALUE)
    facade = create_rule_based_game(knowledge, "tictactoe")
    actions = (Action("place", (("col", 1), ("row", 1))), Action("place", (("col", 2), ("row", 2))))

    assert create_rule_heuristic().rate(rbs, facade.node(facade.start()), actions, "X") == (0.0, 1.0)


def test_a_ruleset_with_no_rule_of_that_kind_says_nothing(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 1.0)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    start = played.node(played.start())

    assert create_rule_heuristic().rate(rbs, start, (Action("place", (("col", 1), ("row", 1))),), "X") == (None,)


def test_two_rulesets_judging_alike_describe_alike(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.5)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)

    described = create_rule_heuristic().describe(rbs)

    assert '"a constant"' in described and "0.5" in described


def test_a_feature_is_extracted_once_and_shared_by_every_model_reading_the_node(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    game("tictactoe")
    heuristic("tictactoe", "my winning moves", PythonRule("wins(me)"), 1.0)
    rbs = create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)
    played = create_rule_based_game(knowledge, "tictactoe")
    node = played.node(played.start())
    reader = create_rule_heuristic()

    reader.value(rbs, node, "X")
    reader.value(rbs, node, "X")

    assert list(node.features) == ["consequences of X"]


def a_heuristic_of(game, knowledge, heuristic, weights) -> object:
    """A position value ruleset of rules that each read a constant, at the weights given."""
    game("tictactoe")
    for name, weight in weights:
        heuristic("tictactoe", name, PythonRule("1.0"), weight)
    return create_rule_based_system(knowledge, "tictactoe", POSITION_VALUE)


def test_every_rule_is_read_unless_a_caller_asks_for_less(game, knowledge, heuristic) -> None:
    """One is what this did before, and a run that never asked for a share must value exactly as it valued."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (("heavy", 1.0), ("light", 0.001)))

    assert len(create_rule_heuristic()._weighted(rbs, POSITION)) == 2  # noqa: SLF001


def test_a_tail_too_light_to_matter_is_not_read_for_this_decision(game, knowledge, heuristic) -> None:
    """**Measured on a real fitted heuristic**: three rules of eight carried the whole weight, the other five
    between 0.00072 and 0.000036 against 0.494. Reading to 99.9% of the weight reads three instead of six here
    and cannot shift a value by more than a thousandth of what the ruleset can say.

    The rules are read heaviest first, which is the order a fit already writes them in."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (
        ("first", 0.494), ("second", -0.494), ("third", 0.501),
        ("a whisper", 0.00072), ("quieter", 0.00066), ("quietest", 0.000036),
    ))

    read = create_rule_heuristic(reading=0.999)._weighted(rbs, POSITION)  # noqa: SLF001

    assert [rule.name for rule, _ in read] == ["third", "first", "second"], "heaviest first, and only those"


def test_what_is_left_unread_stays_in_the_ruleset(game, knowledge, heuristic) -> None:
    """The rules are not dropped. A rule too quiet to matter in this position may be the rule that decides
    another one, and taking it out for being quiet here is a rule dropped for being specific — which is the
    error this project has a standing rule against."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (("heavy", 1.0), ("light", 0.0001)))

    read = create_rule_heuristic(reading=0.9)._weighted(rbs, POSITION)  # noqa: SLF001

    assert len(read) == 1, "only the heavy one was read here"
    assert len(rbs.rules) == 2, "and the ruleset is untouched, so the next position asks the whole set again"


def test_a_ruleset_whose_weights_are_all_nothing_is_read_whole(game, knowledge, heuristic) -> None:
    """There is no heaviest, so there is nothing to leave out."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (("one", 0.0), ("another", 0.0)))

    assert len(create_rule_heuristic(reading=0.5)._weighted(rbs, POSITION)) == 2  # noqa: SLF001


def test_a_valuing_given_no_clock_reads_every_rule(game, knowledge, heuristic) -> None:
    """Nought is no limit, which is what a caller that asks for nothing gets."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (("one", 1.0), ("another", 0.5)))
    played = create_rule_based_game(knowledge, "tictactoe")

    assert create_rule_heuristic().value(rbs, played.node(played.start()), "X") == 1.5


def test_a_valuing_out_of_time_stops_and_keeps_what_it_read(game, knowledge, heuristic) -> None:
    """**The constraint the search actually has.** A node gets so many seconds and values every successor
    within it, so a rule that cannot be afforded here is left for a format that can afford it. Measured: one
    look-ahead rule cost 426 ms a position where every other rule cost 0.2 ms.

    The heaviest rule is always read, so a budget too small for anything still says what mattered most."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (("heavy", 1.0), ("light", 0.5), ("lighter", 0.25)))
    played = create_rule_based_game(knowledge, "tictactoe")
    start = played.node(played.start())

    value = create_rule_heuristic(seconds=1e-9).value(rbs, start, "X")

    assert value == 1.0, "the heaviest rule, and nothing it had no time for"


def test_the_clock_is_checked_between_rules_and_never_inside_one(game, knowledge, heuristic) -> None:
    """A rule half read is worth nothing, so one that has started is finished. What a budget bounds is how
    many rules are read, not how long any one of them may take."""
    rbs = a_heuristic_of(game, knowledge, heuristic, (("only", 2.0),))
    played = create_rule_based_game(knowledge, "tictactoe")

    assert create_rule_heuristic(seconds=1e-12).value(rbs, played.node(played.start()), "X") == 2.0
