import pickle

from openmind.agent.model.domain import Domain
from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.constant.consequence_constant import OUTSIDE
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


def strip_domain() -> Domain:
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
    return Domain(
        "strip",
        position({}, "X"),
        Problem((ActionDefinition("place", (Variable("col", DiscreteDomain((1, 2, 3, 4))),), constraints),)),
        TransitionModel((Transition("place", (Branch(1.0, effects),)),)),
        Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)")),
    )


def position(marks: dict[int, str], turn: str) -> State:
    variables = {**{f"cell(1,{col})": marks.get(col) for col in range(1, 5)}, "payoff(O)": None, "payoff(X)": None, "turn": turn}
    return State(tuple(sorted(variables.items())))


def place(col: int) -> Action:
    return Action("place", (("col", col),))


def coin_domain() -> Domain:
    """A made-up game of chance: A flips a coin, heads A wins, tails B wins."""
    heads = Branch(0.5, PythonRule("payoff['A'] = 1.0\npayoff['B'] = 0.0"))
    tails = Branch(0.5, PythonRule("payoff['A'] = 0.0\npayoff['B'] = 1.0"))
    return Domain(
        "coin",
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A"))),
        Problem((ActionDefinition("flip", (), (PythonRule("payoff['A'] is None"),)),)),
        TransitionModel((Transition("flip", (heads, tails)),)),
        Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
    )


def test_me_and_other_are_the_player_to_act_and_the_next_player() -> None:
    names = ConsequenceLibraryBuilder().build().names(strip_domain(), position({1: "X"}, "O"))

    assert (names["me"], names["other"]) == ("O", "X")


def test_win_chance_is_one_for_a_move_that_wins_and_zero_otherwise() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "X")

    assert (library.win_chance(domain, state, place(2)), library.win_chance(domain, state, place(3))) == (1.0, 0.0)


def test_win_chance_is_the_probability_of_the_winning_outcomes() -> None:
    domain = coin_domain()

    assert ConsequenceLibraryBuilder().build().win_chance(domain, domain.initial_state, Action("flip", ())) == 0.5


def test_wins_counts_what_a_player_would_win_with_if_it_were_their_turn() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "O")

    assert (library.wins(domain, state, "X"), library.wins(domain, state, "O")) == (1.0, 0.0)


def test_wins_after_an_action_is_read_in_the_state_it_leads_to() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "O")

    assert (library.wins(domain, state, "X", place(2)), library.wins(domain, state, "X", place(4))) == (0.0, 1.0)


class CountingSolver:
    """A solver counting how often it is asked to solve."""

    def __init__(self) -> None:
        self._solver = SolverBuilder().build()
        self.calls = 0

    def solve(self, problem: Problem, state: State) -> tuple[Action, ...]:
        self.calls += 1
        return self._solver.solve(problem, state)


def test_a_win_is_a_finished_position_where_the_player_scored_highest() -> None:
    domain = strip_domain()
    marks = dict(position({1: "X", 2: "X"}, "O").variables)
    finished = State(tuple(sorted({**marks, "payoff(X)": 1.0, "payoff(O)": 0.0}.items())))
    library = ConsequenceLibraryBuilder().build()
    x, o = domain.players.names.index("X"), domain.players.names.index("O")

    assert (library.is_win(domain, finished, x), library.is_win(domain, finished, o)) == (True, False)


def test_a_position_with_an_unset_payoff_is_no_win_without_solving() -> None:
    domain, solver = strip_domain(), CountingSolver()
    library = ConsequenceLibrary(solver, PredictorBuilder().build(), StateReader(), VariableNameMapper())  # type: ignore[arg-type]

    assert library.is_win(domain, position({1: "X"}, "O"), domain.players.names.index("X")) is False
    assert solver.calls == 0


def test_a_copy_sent_to_another_process_leaves_its_lookups_behind() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "O")
    library.names(domain, state)

    copy = pickle.loads(pickle.dumps(library))

    assert copy.wins(domain, state, "X") == 1.0


def test_solo_distance_counts_own_moves_until_a_win_is_possible() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "O")

    assert library.solo_distance(domain, state, "X") == 1
    assert library.solo_distance(domain, state, "O") == 2
    assert library.solo_distance(domain, state, "O", limit=1) == 2


def test_near_reads_the_variable_at_an_offset_from_the_one_the_action_sets() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "X")

    assert [library.near(domain, state, place(2), offset) for offset in ((0, -1), (0, 1), (0, -2), (1, 0), (1,))] == [
        "X",
        None,
        OUTSIDE,
        OUTSIDE,
        OUTSIDE,
    ]


def test_rules_read_the_consequences_through_the_names() -> None:
    library, domain, state = ConsequenceLibraryBuilder().build(), strip_domain(), position({1: "X"}, "X")
    rule = RuleCompiler().compile_value(
        PythonRule("win_chance(action) == 1 and near(action, 0, -1) == me and wins(other) == 0 and solo_distance(me) == 1"),
        ("col", "action"),
    )

    assert RuleRunner(StateNamespaceMapper(VariableNameMapper())).value(
        rule, state, {"col": 2, "action": place(2)}, library.names(domain, state)
    ) is True
