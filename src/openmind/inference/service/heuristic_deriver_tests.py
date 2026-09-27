from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.example import Example
from openmind.inference.model.signature import Signature
from openmind.inference.model.heuristic import OPTIMAL
from openmind.inference.model.vocabulary import Vocabulary
from openmind.inference.model.worth import Worth
from openmind.inference.service.heuristic_deriver import HeuristicDeriver
from openmind.knowledge.constant.rule_kind_constant import POSITION
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Variable

LEGAL = Literal("legal", ())


def rule(*body: Literal) -> Clause:
    return Clause((LEGAL, *(one.denied for one in body)))


def about(kind: str) -> Literal:
    return Literal("at", (Constant("piece"), Constant("source"), Constant(kind)))


def steps(value: object) -> Literal:
    return Literal("steps", (Constant("source"), Constant("target"), Constant(value) if not isinstance(value, Variable) else value))


def way(kind: str, distance: int) -> Example:
    """One way of acting: a thing of that kind moving that far."""
    return Example((about(kind), Literal("steps", (Constant("source"), Constant("target"), Constant(distance)))), True)


#: Every way of acting that was ever seen — what a rule is counted against.
WAYS = tuple(way(kind, distance) for kind in ("walker", "runner") for distance in range(1, 8))


def test_it_says_what_the_rules_are_about_without_being_told_what_to_look_for() -> None:
    clauses = (rule(about("walker"), steps(1)), rule(about("runner"), steps(Variable("N"))))

    found = HeuristicDeriver().derive(clauses, WAYS, InferenceBudget(5.0))

    assert {one.about for one in found} == {"walker", "runner"}


def test_a_thing_whose_rules_admit_more_is_worth_more() -> None:
    clauses = (rule(about("walker"), steps(1)), rule(about("runner"), steps(Variable("N"))))

    found = {one.about: one.value for one in HeuristicDeriver().derive(clauses, WAYS, InferenceBudget(5.0))}

    assert found["runner"] > found["walker"]


def test_a_rule_pinning_everything_down_affords_the_one_way_it_pins_down() -> None:
    clauses = (rule(about("walker"), steps(1)),)

    found = HeuristicDeriver().derive(clauses, WAYS, InferenceBudget(5.0))

    assert found[0].value == 1.0


def test_what_a_thing_affords_is_a_count_of_ways_that_exist_and_never_more_than_there_are() -> None:
    """Several readings of one and the same acting are not several choices. Counted as though they were — the
    product of how many values each takes — a rule admitting a handful of ways comes out in the thousands, which
    is not a count of anything. Counted over the ways there are, it can never exceed them."""
    clauses = (rule(about("runner"), steps(Variable("N"))),)

    found = HeuristicDeriver().derive(clauses, WAYS, InferenceBudget(5.0))

    assert found[0].value <= len(WAYS)


def test_a_value_comes_back_with_the_rules_it_was_reasoned_from() -> None:
    clauses = (rule(about("walker"), steps(1)),)

    found = HeuristicDeriver().derive(clauses, WAYS, InferenceBudget(5.0))

    assert found[0].derivation.steps[0].clause == clauses[0]
    assert found[0].derivation.steps[-1].rule == "resolved"


def test_a_value_is_a_starting_point_carrying_where_it_came_from() -> None:
    clauses = (rule(about("walker"), steps(1)),)

    found = HeuristicDeriver().derive(clauses, WAYS, InferenceBudget(5.0))

    assert found[0].parameters[0].holds == "what walker affords"


def test_what_one_thing_may_do_taking_in_what_another_may_do_is_concluded_without_counting() -> None:
    anyhow = rule(about("runner"))
    narrowly = rule(about("walker"), steps(1))

    ordered = HeuristicDeriver().ordered((anyhow, narrowly))

    assert ("runner", "walker") in ordered


def test_things_whose_rules_take_in_each_other_are_not_ordered() -> None:
    one = rule(about("walker"), steps(1))
    other = rule(about("runner"), steps(1))

    assert HeuristicDeriver().ordered((one, other)) == ()


def test_rules_about_nothing_in_particular_yield_nothing_to_value() -> None:
    assert HeuristicDeriver().derive((rule(steps(1)),), WAYS, InferenceBudget(5.0)) == ()


def refuses_a_walker_going_far(clause: Clause, case: Example) -> bool:
    """A stand-in for putting a constraint to a case: this one refuses a walker moving more than one."""
    del clause
    said = {one.predicate: one.arguments[-1].name for one in case.literals}
    return said.get("at") == "walker" and int(said.get("steps", 0)) > 1


def test_what_a_thing_affords_under_refusals_is_what_survives_and_not_what_is_forbidden() -> None:
    """The same question asked of rules that say what is refused. A walker held to one step affords one way
    where a runner affords seven — and counting the cases the *rules match* instead would make the walker the
    richer of the two, since it is the one the rules are about. That inversion is the whole reason this is not
    `derive` with different arguments."""
    constraints = (rule(about("walker"), steps(2)),)

    found = {one.about: one.parameters[0].initial for one in
             HeuristicDeriver().afforded(constraints, WAYS, refuses_a_walker_going_far)}

    assert found["walker"] == 1.0
    assert found["runner"] == 7.0


def test_a_thing_no_rule_mentions_is_still_priced_because_the_cases_carry_it() -> None:
    """A refusal is usually about no particular thing — "you may not move off the board" names none — so
    grouping by what the rules pin down would find nothing to price. What a case is about is in the case."""
    found = HeuristicDeriver().afforded((), WAYS, lambda clause, case: False)

    assert {one.about for one in found} >= {"walker", "runner"}
    assert all(one.parameters[0].initial == 7.0 for one in found if one.about in ("walker", "runner"))


def test_a_derived_heuristic_says_both_what_rule_it_becomes_and_what_it_aims_at():
    """`kind` is the rule kind it becomes in the knowledge base; `aim` is whether it was derived for optimal or
    predictive play. Both were `optimal` because the aim was passed into the kind's slot, so nothing reading a
    heuristic could tell where to file it — and the two must never be confused, since optimal and predictive are
    fitted against different targets and a model fitted against the other's is neither."""
    found = HeuristicDeriver().derive((rule(about("walker"), steps(1)),), WAYS, InferenceBudget(seconds=1.0))

    assert found
    assert all(one.kind == POSITION for one in found)
    assert all(one.aim == OPTIMAL for one in found)


def a_vocabulary(values_by_base, players=("white", "black"), places=((0, 0), (0, 1), (1, 0), (1, 1))):
    """A vocabulary of bases over the same places, as the expression generator reads one."""
    return Vocabulary(
        players,
        {},
        dict(values_by_base),
        {base: frozenset(places) for base in values_by_base},
        {},
        grids=frozenset(values_by_base),
    )


def worth_of(model, value, held):
    return Worth(((model, value, held),), ended=4, settled=True)


def test_a_worth_becomes_a_heuristic_that_knows_which_structure_its_thing_stands_in():
    """Everything else gives back a bare value — a knight, with nothing saying which of a position's structures
    a knight stands in — and an expression cannot be built from that without guessing."""
    found = HeuristicDeriver().holdings(worth_of("piece", "knight", 14.0))

    assert [one.about for one in found] == ["knight"]
    assert found[0].parameters[0].holds == "piece"
    assert found[0].parameters[0].initial == 14.0
    assert found[0].kind == POSITION


def test_a_worth_the_positions_did_not_bear_out_says_what_it_is_leaning_on():
    """`WorthReasoner` counts the worths anyway where the positions did not show that being able to do less is
    being worse off, and reports them as resting on nothing. That is a premise holding only sometimes, which is
    what `chances` is for — and it had never been populated by anything."""
    unsettled = Worth((("piece", "knight", 14.0),), ended=4, settled=False)

    assert not HeuristicDeriver().holdings(worth_of("piece", "knight", 14.0))[0].derivation.chances
    assert HeuristicDeriver().holdings(unsettled)[0].derivation.chances


def test_a_thing_and_a_base_saying_whose_it_is_seed_the_term_whose_weight_is_what_it_is_worth():
    """Counting knights alone counts both sides' knights, which barely moves between positions and explains
    nothing. The term whose weight *is* what a knight is worth needs the second condition, and the search
    reaches it only by growing a child of the near-constant parent it already passed over."""
    vocabulary = a_vocabulary({"piece": ("knight", "rook"), "color": ("white", "black")})

    found = HeuristicDeriver().seeds(HeuristicDeriver().holdings(worth_of("piece", "knight", 14.0)), vocabulary)

    templates = {one.template for one, _ in found}
    assert "sum(1 for at in {view}.piece if {view}.piece[at] == 'knight')" in templates
    assert "sum(1 for at in {view}.piece if {view}.piece[at] == 'knight' and {view}.color[at] == me)" in templates
    assert all(weight == 14.0 for _, weight in found)


def test_a_game_whose_things_are_the_players_themselves_needs_no_second_base():
    """Tic-tac-toe keeps marks and owners in one grid: a cell holds `X`, and `X` is a player. So the value is
    written as `me` and the one condition already says whose it is."""
    vocabulary = a_vocabulary({"cell": ("X", "O")}, players=("X", "O"))

    found = HeuristicDeriver().seeds(HeuristicDeriver().holdings(worth_of("cell", "X", -1.0)), vocabulary)

    assert {one.template for one, _ in found} == {
        "sum(1 for at in {view}.cell if {view}.cell[at] == me)",
        "sum(1 for at in {view}.cell if {view}.cell[at] == other)",
    }


def test_a_thing_worth_nothing_is_not_seeded():
    """The prisoner's dilemma owns nothing on the table, so every worth is nought and there is no term to
    start the search from. Seeding one would be putting a number where the rules said there was none."""
    vocabulary = a_vocabulary({"played": ("cooperate", "defect")})

    assert not HeuristicDeriver().seeds(HeuristicDeriver().holdings(worth_of("played", "defect", 0.0)), vocabulary)


def test_a_game_with_no_structures_at_all_is_seeded_with_nothing():
    """Rock paper scissors has no grid, so nothing was ever read standing anywhere and there is nothing to
    count. Producing a confident term here would be producing one from no evidence."""
    assert not HeuristicDeriver().seeds(
        HeuristicDeriver().holdings(worth_of("chosen", "rock", 3.0)), a_vocabulary({})
    )
