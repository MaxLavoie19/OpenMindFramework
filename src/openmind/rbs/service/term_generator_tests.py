from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.rbs.service.term_generator import TermGenerator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

SETTINGS = ValueSettings(pair_pool=10, cuts=6, solo_limit=1, prices=(0.1, 0.01, 0.001), max_steps=2000, tolerance=1e-6)


def new_generator() -> TermGenerator:
    names = VariableNameMapper()
    return TermGenerator(new_evaluator(), StateNamespaceMapper(names), names)


def test_terms_read_variables_counts_and_quantities_relative_to_the_player_valued() -> None:
    rows = (
        PositionRow(position({1: "X"}, "O"), "X", 1.0),
        PositionRow(position({1: "X", 2: "O"}, "X"), "O", 0.0),
        PositionRow(position({}, "X"), "X", 0.5),
    )

    terms = [term.source for term in new_generator().generate(strip_domain(), rows, SETTINGS)]

    assert {
        "cell[1, 1] == me",
        "cell[1, 1] == other",
        "cell[1, 1] == None",
        "payoff['X'] == None",
        "turn == 'O'",
        "turn == 'X'",
        "turn == me",
        "turn == other",
        "sum(value == me for value in cell.values())",
        "sum(value == other for value in cell.values())",
        "sum(value == None for value in cell.values())",
        "sum(value == None for value in payoff.values())",
        "wins(me)",
        "wins(me) >= 1",
        "wins(other)",
        "solo_distance(me, None, 1)",
        "solo_distance(other, None, 1)",
    } <= set(terms)
    assert len(terms) == len(set(terms))
    assert not any("'X'" in term or "'O'" in term for term in terms if not term.startswith(("turn", "payoff")))


def test_without_a_solo_limit_solo_distance_is_left_out() -> None:
    rows = (PositionRow(position({1: "X"}, "O"), "X", 1.0),)

    terms = new_generator().generate(strip_domain(), rows, ValueSettings(10, 6, 0, (0.01,), 100, 1e-6))

    assert not any(term.source.startswith("solo_distance") for term in terms)
