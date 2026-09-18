from collections.abc import Callable
import pickle

from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.inference.service.mechanics import Mechanics
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.constant.consequence_constant import OUTSIDE
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


type Declare = Callable[..., RuleBasedSystem]


def strip_domain(declared: Declare) -> RuleBasedSystem:
    """A made-up game: X and O take turns marking a cell of a 1 by 4 strip; two marks side by side win, a full strip is a
    draw."""
    effects = PythonRule(
        "cell[1, col] = turn\n"
        "if any(cell[1, c] == turn and cell[1, c + 1] == turn for c in range(1, 4)):\n"
        "    payoff[turn], payoff['O' if turn == 'X' else 'X'] = 1.0, 0.0\n"
        "elif all(mark is not None for mark in cell.values()):\n"
        "    payoff['X'] = payoff['O'] = 0.5\n"
        "turn = 'O' if turn == 'X' else 'X'"
    )
    constraints = (PythonRule("payoff['X'] is None"), PythonRule("payoff['O'] is None"), PythonRule("cell[1, col] is None"))
    return declared(
        position({}, "X"),
        legal={"place": constraints},
        outcomes={"place": ((1.0, effects),)},
        players=Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)")),
        parameters={"place": {"col": PythonRule("(1, 2, 3, 4)")}},
        empties={"cell": None},
        context="strip",
    )


def position(marks: dict[int, str], turn: str) -> State:
    variables = {**{f"cell(1,{col})": marks.get(col) for col in range(1, 5)}, "payoff(O)": None, "payoff(X)": None, "turn": turn}
    return State(tuple(sorted(variables.items())))


def place(col: int) -> Action:
    return Action("place", (("col", col),))


def coin_domain(declared: Declare) -> RuleBasedSystem:
    """A made-up game of chance: A flips a coin, heads A wins, tails B wins."""
    heads = 0.5, PythonRule("payoff['A'] = 1.0\npayoff['B'] = 0.0")
    tails = 0.5, PythonRule("payoff['A'] = 0.0\npayoff['B'] = 1.0")
    return declared(
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A"))),
        legal={"flip": (PythonRule("payoff['A'] is None"),)},
        outcomes={"flip": (heads, tails)},
        players=Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
        context="coin",
    )


def test_me_and_other_are_the_player_to_act_and_the_next_player(declared: Declare) -> None:
    names = ConsequenceLibraryBuilder().build().names(strip_domain(declared), position({1: "X"}, "O"))

    assert (names["me"], names["other"]) == ("O", "X")


def test_win_chance_is_one_for_a_move_that_wins_and_zero_otherwise(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "X")

    assert (library.win_chance(rbs, state, place(2)), library.win_chance(rbs, state, place(3))) == (1.0, 0.0)


def test_win_chance_is_the_probability_of_the_winning_outcomes(declared: Declare) -> None:
    rbs = coin_domain(declared)

    assert ConsequenceLibraryBuilder().build().win_chance(rbs, rbs.start(), Action("flip", ())) == 0.5


def test_wins_counts_what_a_player_would_win_with_if_it_were_their_turn(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "O")

    assert (library.wins(rbs, state, "X"), library.wins(rbs, state, "O")) == (1.0, 0.0)


def test_wins_after_an_action_is_read_in_the_state_it_leads_to(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "O")

    assert (library.wins(rbs, state, "X", place(2)), library.wins(rbs, state, "X", place(4))) == (0.0, 1.0)


class Counting:
    """An RBS counting how often it is asked for a position's legal moves."""

    def __init__(self, rbs: RuleBasedSystem) -> None:
        self._rbs = rbs
        self.calls = 0

    def actions(self, state: State, limit: int | None = None, player: str | None = None) -> tuple[Action, ...]:
        self.calls += 1
        return self._rbs.actions(state, limit, player)

    def __getattr__(self, name: str) -> object:
        return getattr(self._rbs, name)


def test_a_win_is_a_finished_position_where_the_player_scored_highest(declared: Declare) -> None:
    rbs = strip_domain(declared)
    marks = dict(position({1: "X", 2: "X"}, "O").variables)
    finished = State(tuple(sorted({**marks, "payoff(X)": 1.0, "payoff(O)": 0.0}.items())))
    library = ConsequenceLibraryBuilder().build()
    x, o = rbs.players().names.index("X"), rbs.players().names.index("O")

    assert (library.is_win(rbs, finished, x), library.is_win(rbs, finished, o)) == (True, False)


def test_a_position_with_an_unset_payoff_is_no_win_without_asking_for_its_moves(declared: Declare) -> None:
    counting = Counting(strip_domain(declared))
    mechanics = Mechanics(StateNamespaceMapper(VariableNameMapper()), MemoryMeter())
    library = ConsequenceLibrary(StateReader(), VariableNameMapper(), mechanics)

    assert library.is_win(counting, position({1: "X"}, "O"), counting.players().names.index("X")) is False  # type: ignore[arg-type]
    assert counting.calls == 0


def test_a_copy_sent_to_another_process_leaves_its_lookups_behind(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "O")
    library.names(rbs, state)

    copy = pickle.loads(pickle.dumps(library))

    assert copy.wins(rbs, state, "X") == 1.0


def test_near_reads_the_variable_at_an_offset_from_the_one_the_action_sets(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "X")

    assert [library.near(rbs, state, place(2), offset) for offset in ((0, -1), (0, 1), (0, -2), (1, 0), (1,))] == [
        "X",
        None,
        OUTSIDE,
        OUTSIDE,
        OUTSIDE,
    ]


def test_names_can_bind_me_to_the_player_a_position_is_valued_for(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "X")

    valued_for_o = library.names(rbs, state, "O")

    assert (valued_for_o["me"], valued_for_o["other"]) == ("O", "X")
    assert (library.names(rbs, state)["me"], library.names(rbs, state, "X")["me"]) == ("X", "X")
    assert library.names(rbs, state, "O") is valued_for_o


def test_rules_read_the_consequences_through_the_names(declared: Declare) -> None:
    library, rbs, state = ConsequenceLibraryBuilder().build(), strip_domain(declared), position({1: "X"}, "X")
    rule = RuleCompiler().compile_value(
        PythonRule("win_chance(action) == 1 and near(action, 0, -1) == me and wins(other) == 0"),
        ("col", "action"),
    )

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(
        rule, state, {"col": 2, "action": place(2)}, library.names(rbs, state)
    ) is True
