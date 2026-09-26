import pickle
from collections.abc import Callable

from openmind.inference.constant.inference_constant import HERE, ME, OTHER
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]

MIDDLE = Action("place", (("col", 2), ("row", 2)))


def test_who_is_acting_is_read_off_the_position_where_only_one_player_can(game: Game) -> None:
    """A rule saying *my* mark has to know whose turn it is, and no game tells OMF which of its models says so —
    it falls out of who has an action."""
    played = game("tictactoe")

    names = ConsequenceLibraryBuilder().build().names(played, played.start())

    assert names[ME] == "X"
    assert names[OTHER] == "O"


def test_a_position_valued_for_a_player_is_valued_for_that_player_whoever_is_to_move(game: Game) -> None:
    """A heuristic values a position for each player in turn, including the one not to move, so `me` is the
    player asked about rather than the player acting."""
    played = game("tictactoe")

    names = ConsequenceLibraryBuilder().build().names(played, played.start(), "O")

    assert names[ME] == "O"
    assert names[OTHER] == "X"


def test_the_same_position_and_player_give_the_same_names_back(game: Game) -> None:
    """Every candidate reading of a position asks for these, so building them anew per reading is the
    difference between reading a position and reading it thousands of times."""
    played = game("tictactoe")
    library = ConsequenceLibraryBuilder().build()

    assert library.names(played, played.start()) is library.names(played, played.start())


def test_what_a_rule_reads_of_the_position_comes_with_the_names(game: Game) -> None:
    """A generated term is source over `here`, so a term handed names without one reads nothing at all."""
    played = game("tictactoe")

    assert HERE in ConsequenceLibraryBuilder().build().names(played, played.start())


def test_an_action_that_wins_outright_is_worth_the_whole_chance(game: Game) -> None:
    """A win is an outcome with no legal action left where the player's payoff beats every other's — said that
    way, nothing here knows what winning is in any particular game."""
    played = game("tictactoe")
    state = played.start()
    for column, row in ((1, 1), (2, 1), (1, 2), (2, 2)):  # X takes two of the first column, O answers beside
        state = played.outcomes(state, Action("place", (("col", column), ("row", row)))).outcomes[0][0]

    chance = ConsequenceLibraryBuilder().build().win_chance(played, state, Action("place", (("col", 1), ("row", 3))))

    assert chance == 1.0


def test_an_action_that_does_not_end_the_game_wins_nothing_yet(game: Game) -> None:
    played = game("tictactoe")

    assert ConsequenceLibraryBuilder().build().win_chance(played, played.start(), MIDDLE) == 0.0


def test_what_stands_beside_where_an_action_plays_is_read_by_the_offset(game: Game) -> None:
    """`near` is how a rule says *the square in front of this one* without naming a square, which is what makes
    it a rule rather than a note about one position."""
    played = game("tictactoe")
    after = played.outcomes(played.start(), MIDDLE).outcomes[0][0]

    library = ConsequenceLibraryBuilder().build()

    assert library.near(played, after, MIDDLE, (0, 0)) == "X"


def test_reading_past_the_edge_says_outside_rather_than_nothing(game: Game) -> None:
    """Nothing is what an empty square holds; outside is what is not a square at all, and a rule that could not
    tell them apart would count the edge of the board as empty."""
    from openmind.inference.constant.inference_constant import OUTSIDE

    played = game("tictactoe")
    after = played.outcomes(played.start(), MIDDLE).outcomes[0][0]

    assert ConsequenceLibraryBuilder().build().near(played, after, MIDDLE, (9, 9)) == OUTSIDE


def test_what_it_remembered_stays_behind_when_it_crosses_to_another_process(game: Game) -> None:
    """Its lookups are keyed by this process's object ids and hold functions, so carrying them would either
    fail to pickle or answer about a game that is not the one asking."""
    played = game("tictactoe")
    library = ConsequenceLibraryBuilder().build()
    library.names(played, played.start())

    carried = pickle.loads(pickle.dumps(library))

    assert library.memory_entries() > 0
    assert carried.memory_entries() == 0


def test_forgetting_what_it_remembered_leaves_it_able_to_answer(game: Game) -> None:
    """The guard clears it whenever the process holds more than its share, mid-run, so an emptied library has
    to go on working rather than being a library that was used once."""
    played = game("tictactoe")
    library = ConsequenceLibraryBuilder().build()
    library.names(played, played.start())

    library.clear_memory()

    assert library.memory_entries() == 0
    assert library.names(played, played.start())[ME] == "X"
