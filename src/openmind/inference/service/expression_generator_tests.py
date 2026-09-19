from openmind.inference.model.expression import Expression
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rbs.model.position_row import PositionRow
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.consequence_library_tests import Declare, position, strip_domain
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.inference.service.position_view_tests import capture_domain, capture_position
from openmind.structure.model.map import Map
from openmind.world.model.players import Players
from openmind.world.model.state import State


def line_domain(declared: Declare) -> RuleBasedSystem:
    """A made-up game without a board: A and B each have a token at a real position on a line; the player to act moves
    their own token 0.75 left or right; tokens closer than 0.5 end the game, a win for the player who moved."""
    effects = PythonRule(
        "x = x.with_item(turn, x[turn] + step)\n"
        "if abs(x['A'] - x['B']) < 0.5:\n"
        "    payoff = Map.of({turn: 1.0, 'B' if turn == 'A' else 'A': 0.0})\n"
        "turn = 'B' if turn == 'A' else 'A'"
    )
    constraints = (PythonRule("payoff['A'] is None"), PythonRule("payoff['B'] is None"))
    return declared(
        line_position(0.0, 2.0, "A"),
        legal={"move": constraints},
        outcomes={"move": ((1.0, effects),)},
        players=Players(("A", "B"), "turn", "payoff"),
        parameters={"move": {"step": PythonRule("(-0.75, 0.75)")}},
        context="line",
    )


def line_position(a: float, b: float, turn: str) -> State:
    return State.of(payoff=Map.of({"A": None, "B": None}), turn=turn, x=Map.of({"A": a, "B": b}))


def strip_vocabulary(declared: Declare) -> Vocabulary:
    states = (position({1: "X"}, "O"), position({1: "X", 2: "O"}, "X"), position({}, "X"))
    return ExpressionGenerator().vocabulary(strip_domain(declared), states)


def test_the_vocabulary_holds_the_values_indices_and_offsets_seen(declared: Declare) -> None:
    vocabulary = strip_vocabulary(declared)

    assert vocabulary.values_by_base["cell"] == ("X", None, "O")
    assert vocabulary.indices_by_base["cell"] == frozenset({(1, 1), (1, 2), (1, 3), (1, 4)})
    assert vocabulary.offsets_by_arity[2] == ((0, -3), (0, -2), (0, -1), (0, 1), (0, 2), (0, 3))
    assert vocabulary.grids == frozenset({"cell"})
    assert vocabulary.values_by_variable[("turn", ())] == ("O", "X")
    assert vocabulary.values_by_variable[("payoff", ("X",))] == (None,)


def test_a_cell_of_a_grid_of_one_dimension_is_read_by_its_coordinates(declared: Declare) -> None:
    generator, rbs = ExpressionGenerator(), capture_domain(declared)
    vocabulary = generator.vocabulary(rbs, (capture_position({1: "A", 7: "B"}, "A"),))

    templates = {leaf.template for leaf in generator.leaves(vocabulary)}

    assert vocabulary.offsets_by_arity[1][:2] == ((-6,), (-5,))
    assert {"{view}.cell[1,] == me", "{view}.cell[7,] == other", "sum(1 for at in {view}.cell if {view}.cell[at] == me)"} <= templates
    rows = [PositionRow(capture_position({1: "A", 7: "B"}, "A"), "A", 0.0)]
    sources = [generator.source(Expression(template, 1, 0)) for template in ("{view}.cell[1,] == me", "{view}.offset('cell', (1,), 6) == other")]
    assert [list(column) for column in new_evaluator().columns(rbs, rows, sources)] == [[1.0], [1.0]]  # type: ignore[arg-type]


def test_leaves_read_every_variable_count_every_value_and_the_players_mobility(declared: Declare) -> None:
    templates = {leaf.template for leaf in ExpressionGenerator().leaves(strip_vocabulary(declared))}

    assert {
        "{view}.cell[1, 1] == me",
        "{view}.cell[1, 1] == other",
        "{view}.cell[1, 1] == None",
        "{view}.turn == 'X'",
        "{view}.turn == me",
        "{view}.payoff['X'] == None",
        "{view}.payoff[me] == None",
        "{view}.payoff[other] == None",
        "sum(1 for at in {view}.cell if {view}.cell[at] == me)",
        "sum(1 for at in {view}.payoff if {view}.payoff[at] == None)",
        "{view}.mobility(me)",
        "{view}.mobility(other)",
    } <= templates
    assert not any("'O'" in template for template in templates if "cell" in template)


def test_a_pattern_grows_by_a_condition_anywhere_around_its_index(declared: Declare) -> None:
    generator, vocabulary = ExpressionGenerator(), strip_vocabulary(declared)
    (mine,) = [leaf for leaf in generator.leaves(vocabulary) if leaf.template == "sum(1 for at in {view}.cell if {view}.cell[at] == me)"]

    children = {child.template: child for child in generator.pattern_children(mine, vocabulary)}

    two_in_a_row = "sum(1 for at in {view}.cell if {view}.cell[at] == me and {view}.offset('cell', at, 0, 1) == me)"
    assert two_in_a_row in children
    assert "sum(1 for at in {view}.cell if {view}.cell[at] == me and {view}.offset('cell', at, 0, -3) != None)" in children
    assert "sum(1 for at in {view}.cell if {view}.cell[at] == me and {view}.offset('cell', at, 0, 2) == '<outside>')" in children
    assert (
        "sum(1 for at in {view}.cell if {view}.cell[at] == me and {view}.offset('cell', at, 0, 1) == {view}.cell[at])"
        in children
    )
    assert (children[two_in_a_row].clauses, children[two_in_a_row].plies) == (2, 0)
    grandchildren = {child.template for child in generator.pattern_children(children[two_in_a_row], vocabulary)}
    assert (
        "sum(1 for at in {view}.cell if {view}.cell[at] == me and {view}.offset('cell', at, 0, 1) == me "
        "and {view}.offset('cell', at, 0, 2) == me)"
    ) in grandchildren


def test_numbers_are_read_as_numbers_and_aggregated_never_compared_with_a_value_seen(declared: Declare) -> None:
    generator, rbs = ExpressionGenerator(), line_domain(declared)
    vocabulary = generator.vocabulary(rbs, (line_position(0.0, 2.0, "A"), line_position(1.25, -0.5, "B")))

    templates = {leaf.template for leaf in generator.leaves(vocabulary)}

    assert {
        "{view}.x['A']",
        "{view}.x[me]",
        "{view}.x[other]",
        "sum(({view}.x[i]) for i in {view}.x)",
        "min((({view}.x[i]) for i in {view}.x), default=0)",
        "max((({view}.x[i]) for i in {view}.x), default=0)",
        "{view}.turn == 'A'",
    } <= templates
    assert not any("== 1.25" in template or "== 2.0" in template or "offset(" in template for template in templates)


def test_an_aggregate_grows_into_relations_between_pairs_of_indices(declared: Declare) -> None:
    generator, rbs = ExpressionGenerator(), line_domain(declared)
    states = (line_position(0.0, 2.0, "A"), line_position(1.0, 0.25, "B"))
    vocabulary = generator.vocabulary(rbs, states)
    (lowest,) = [
        leaf for leaf in generator.leaves(vocabulary) if leaf.template == "min((({view}.x[i]) for i in {view}.x), default=0)"
    ]

    children = {child.template: child for child in generator.aggregate_children(lowest, vocabulary)}

    gap = "min(((abs(({view}.x[i]) - ({view}.x[j]))) for i in {view}.x for j in {view}.x if i != j), default=0)"
    assert gap in children and "sum(1 for i in {view}.x for j in {view}.x if i != j if ({view}.x[i]) >= ({view}.x[j]))" in children
    assert (children[gap].clauses, children[gap].aggregate.pair) == (4, True)  # type: ignore[union-attr]
    grown = {child.template for child in generator.aggregate_children(children[gap], vocabulary)}
    assert (
        "max((((abs(({view}.x[i]) - ({view}.x[j]))) * ({view}.x[j])) for i in {view}.x for j in {view}.x if i != j), default=0)"
        in grown
    )
    rows = [PositionRow(state, "A", 0.0) for state in states]
    (column,) = new_evaluator().columns(rbs, rows, [generator.source(children[gap])])
    assert column is not None and list(column) == [2.0, 0.75]
    assert generator.unary(Expression("{view}.x[me]", 1, 0))[0][0].template == "abs({view}.x[me])"


def count_of(body: str) -> str:
    return f"sum(1 for i in {{view}}.cell if {body})"


def test_a_grid_of_names_counts_its_cells_by_the_actions_changing_them(declared: Declare) -> None:
    generator, vocabulary = ExpressionGenerator(), strip_vocabulary(declared)
    leaves = {leaf.template: leaf for leaf in generator.leaves(vocabulary)}
    fillable = count_of("(({view}.cell[i] == None) and ({view}.changed(me, 'cell', i)))")
    rows = (PositionRow(position({1: "X", 2: "X", 3: "O"}, "O"), "X", 0.0), PositionRow(position({1: "X"}, "O"), "X", 0.0))

    (column,) = new_evaluator().columns(strip_domain(declared), rows, [generator.source(leaves[fillable])])

    assert (leaves[fillable].clauses, leaves[fillable].plies) == (3, 1)
    assert list(column) == [1.0, 3.0]  # type: ignore[arg-type]


def test_an_aggregate_over_a_grid_grows_by_changes_and_what_ifs_at_each_cell_and_pairs_of_cells(declared: Declare) -> None:
    generator, vocabulary = ExpressionGenerator(), strip_vocabulary(declared)
    empty = "(({view}.cell[i] == None) and ({view}.changed(me, 'cell', i)))"
    (fillable,) = [leaf for leaf in generator.leaves(vocabulary) if leaf.template == count_of(empty)]

    children = {child.template: child for child in generator.aggregate_children(fillable, vocabulary)}

    templates = "\n".join(children)
    assert all(
        reading in templates
        for reading in (
            "{view}.changed(other, 'cell', i)",
            "{view}.changed(me, 'cell', j)",
            "{view}.alone(i).mobility(me)",
            "{view}.with_value('cell', i, other).changed(me, 'cell', i)",
            "{view}.with_value('cell', i, me).changed(other, 'cell', i)",
        )
    )
    defended = next(child for child in children.values() if child.template == count_of(f"(({empty}) and ({{view}}.with_value('cell', i, other).changed(me, 'cell', i)))"))
    compared = "\n".join(child.template for child in generator.aggregate_children(defended, vocabulary))
    assert "{view}.with_value('cell', j, other).changed(me, 'cell', j)" in compared
    (pairs,) = [child for child in children.values() if child.template.startswith("sum(1 for i in {view}.cell for j") and "- ((({view}.cell[j]" in child.template][:1]
    grown = "\n".join(child.template for child in generator.aggregate_children(pairs, vocabulary))
    assert all(
        query in grown
        for query in (
            "{view}.copied(j, i).changed(other, 'cell', i)",
        )
    )


def test_thresholds_combinations_and_look_aheads(declared: Declare) -> None:
    generator = ExpressionGenerator()
    marks, mobility = Expression("{view}.marks", 1, 0), Expression("{view}.best(me, lambda v1: v1.marks)", 2, 1)

    assert [(child.template, relation, cut) for child, relation, cut in generator.thresholds(marks, [2.0, 0.0, 2.0, 1.5])] == [
        ("({view}.marks) >= 1.5", ">=", 1.5),
        ("({view}.marks) >= 2", ">=", 2.0),
        ("({view}.marks) <= 0", "<=", 0.0),
        ("({view}.marks) <= 1.5", "<=", 1.5),
    ]
    combined = {operator: child for child, operator in generator.combinations(marks, mobility)}
    assert combined["/"].template == "({view}.marks) / max(1, {view}.best(me, lambda v1: v1.marks))"
    assert (combined["/"].clauses, combined["/"].plies) == (3, 1)
    looks = [child.template for child in generator.look_aheads(mobility)]
    assert "{view}.worst(other, lambda v2: v2.best(me, lambda v1: v1.marks))" in looks
    assert "{view}.best(me, lambda v2: (v2.best(me, lambda v1: v1.marks)) - ({view}.best(me, lambda v1: v1.marks)))" in looks
    assert "{view}.count(other, lambda v2: (v2.best(me, lambda v1: v1.marks)) > ({view}.best(me, lambda v1: v1.marks)))" in looks
    assert len(looks) == 14 and all(child.plies == 2 for child in generator.look_aheads(mobility))


def test_sources_read_here_and_evaluate_on_positions(declared: Declare) -> None:
    generator = ExpressionGenerator()
    pattern = Expression("sum(1 for at in {view}.cell if {view}.cell[at] == me and {view}.offset('cell', at, 0, 1) == me)", 2, 0)
    threat = Expression("{view}.count(me, lambda v1: v1.payoff[me] == 1.0)", 2, 1)
    rows = (
        PositionRow(position({1: "X", 2: "X", 3: "O", 4: "O"}, "O"), "X", 1.0),
        PositionRow(position({1: "X", 3: "O"}, "X"), "X", 0.0),
    )

    assert generator.source(threat).source == "here.count(me, lambda v1: v1.payoff[me] == 1.0)"
    columns = new_evaluator().columns(strip_domain(declared), rows, [generator.source(pattern), generator.source(threat)])
    assert [list(column) for column in columns] == [[1.0, 0.0], [0.0, 1.0]]  # type: ignore[arg-type]
