from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.example import Example
from openmind.inference.model.signature import Signature
from openmind.inference.service.heuristic_deriver import HeuristicDeriver
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Variable

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
