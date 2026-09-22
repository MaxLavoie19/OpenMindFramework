from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.clause_learner import ClauseLearner
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Variable

LEGAL = Literal("legal", ())


def case(holds: bool, where: object = None, **readings: object) -> Example:
    """One case, its readings written as `predicate=value` or `predicate=(value, value)`."""
    literals = tuple(
        Literal(name, tuple(Constant(one) for one in (value if isinstance(value, tuple) else (value,))))
        for name, value in readings.items()
    )
    return Example(literals, holds, where)


def learned(examples: list[Example], **kept: object) -> tuple:
    return ClauseLearner().learn(examples, LEGAL, InferenceBudget(10.0), **kept)  # type: ignore[arg-type]


def test_one_reading_telling_the_cases_apart_is_learned_as_one_clause() -> None:
    examples = [
        case(True, kind="walker", step=1),
        case(True, kind="walker", step=1),
        case(False, kind="walker", step=5),
        case(False, kind="walker", step=9),
    ]

    found = learned(examples)

    assert len(found) == 1 and Literal("step", (Constant(1),)) in found[0].body


def test_a_clause_generalised_from_cases_keeps_what_they_all_agreed_on_until_it_is_relaxed() -> None:
    """Generalising gives the least general clause covering the cases, which keeps readings they happen to share
    whether or not those readings rule anything out. Taking the surplus back out is what relaxing is for, and it
    needs cases that hold and are turned away to see which readings are surplus."""
    learner = ClauseLearner()
    examples = [
        case(True, kind="walker", step=1),
        case(True, kind="walker", step=1),
        case(False, kind="walker", step=5),
    ]
    grown = learner.learn(examples, LEGAL, InferenceBudget(10.0))

    wider = [*examples, case(True, kind="flyer", step=1)]
    relaxed = learner.relaxed(grown, wider)

    assert Literal("kind", (Constant("walker"),)) in grown[0].body
    assert Literal("kind", (Constant("walker"),)) not in relaxed[0].body


def test_what_is_learned_covers_the_cases_it_was_learned_from() -> None:
    examples = [
        case(True, kind="walker", step=1),
        case(False, kind="walker", step=5),
    ]

    found = learned(examples)

    assert ClauseLearner().covers(found[0], examples[0])
    assert not ClauseLearner().covers(found[0], examples[1])


def test_two_ways_of_holding_are_learned_as_two_clauses_rather_than_one() -> None:
    examples = [
        case(True, kind="walker", step=1),
        case(True, kind="walker", step=1),
        case(True, kind="jumper", step=3),
        case(True, kind="jumper", step=3),
        case(False, kind="walker", step=3),
        case(False, kind="jumper", step=1),
        case(False, kind="walker", step=7),
    ]

    found = learned(examples)

    assert len(found) == 2
    assert all(ClauseLearner().covered(found, one) for one in examples if one.holds)
    assert not any(ClauseLearner().covered(found, one) for one in examples if not one.holds)


def test_learning_stops_when_every_case_is_accounted_for_and_not_at_a_count() -> None:
    examples = [case(True, kind=name, step=1) for name in ("a", "b", "c", "d", "e")]
    examples += [case(False, kind="a", step=2)]

    found = learned(examples)

    assert len(found) == 1


def test_a_variable_ties_two_readings_together_so_one_clause_says_what_two_would() -> None:
    examples = [
        case(True, acting="first", owner="first", step=1),
        case(True, acting="second", owner="second", step=1),
        case(False, acting="first", owner="second", step=1),
        case(False, acting="second", owner="first", step=1),
    ]

    found = learned(examples)

    assert len(found) == 1
    body = found[0].body
    assert any(isinstance(one.arguments[0], Variable) for one in body)


def test_the_clause_tying_two_readings_covers_both_players_where_conditions_on_values_would_need_two() -> None:
    examples = [
        case(True, acting="first", owner="first", step=1),
        case(True, acting="second", owner="second", step=1),
        case(False, acting="first", owner="second", step=1),
        case(False, acting="second", owner="first", step=1),
    ]

    found = learned(examples)
    learner = ClauseLearner()

    assert learner.covers(found[0], examples[0]) and learner.covers(found[0], examples[1])
    assert not learner.covers(found[0], examples[2]) and not learner.covers(found[0], examples[3])


def test_what_no_clause_accounts_for_is_counted_by_the_reading_asked_about() -> None:
    examples = [
        case(True, kind="walker", step=1),
        case(True, kind="flyer", step=4),
        case(False, kind="walker", step=5),
    ]
    learner = ClauseLearner()
    found = learner.learn(examples[:1] + examples[2:], LEGAL, InferenceBudget(10.0))

    missing = learner.uncovered(found, examples, "kind")

    assert missing.get("flyer") == 1


def test_the_signature_is_read_off_the_cases_and_holds_nothing_they_did_not_offer() -> None:
    examples = [case(True, kind="walker", step=1), case(False, kind="flyer", step=4)]

    signature = ClauseLearner().signature(examples)

    assert signature.arity("kind") == 1
    assert set(signature.seen("kind", 0)) == {"walker", "flyer"}
    assert signature.arity("nothing anyone read") is None


def test_a_reading_whose_every_value_was_a_number_is_marked_numeric() -> None:
    examples = [case(True, step=1), case(False, step=4)]

    signature = ClauseLearner().signature(examples)

    assert signature.is_numeric("step", 0)


def test_a_reading_of_names_is_not_marked_numeric() -> None:
    examples = [case(True, kind="walker"), case(False, kind="flyer")]

    assert not ClauseLearner().signature(examples).is_numeric("kind", 0)


def test_cases_nothing_read_of_them_tells_apart_are_learned_from_as_far_as_they_can_be() -> None:
    examples = [case(True, kind="walker"), case(False, kind="walker")]

    found = learned(examples)

    assert found == ()


def test_a_clause_is_never_kept_with_no_conditions_at_all() -> None:
    examples = [case(True, kind="walker"), case(True, kind="flyer")]

    found = learned(examples)

    assert all(one.body for one in found)


def test_a_condition_that_turns_away_cases_which_hold_is_dropped_as_an_artefact() -> None:
    learner = ClauseLearner()
    grown = learner.learn(
        [case(True, kind="walker", step=1), case(False, kind="flyer", step=1)], LEGAL, InferenceBudget(10.0)
    )
    wider = [case(True, kind="walker", step=1), case(True, kind="walker", step=2), case(False, kind="flyer", step=1)]

    relaxed = learner.relaxed(grown, wider)

    assert all(learner.covers(relaxed[0], one) for one in wider if one.holds)


def test_a_condition_that_rules_something_out_is_kept_when_relaxing() -> None:
    learner = ClauseLearner()
    examples = [case(True, kind="walker"), case(False, kind="flyer")]
    grown = learner.learn(examples, LEGAL, InferenceBudget(10.0))

    relaxed = learner.relaxed(grown, examples)

    assert relaxed[0].body and not learner.covers(relaxed[0], examples[1])


def test_what_a_clause_wrongly_covers_is_answered_for_by_an_exclusion_of_its_own() -> None:
    learner = ClauseLearner()
    examples = [
        case(True, kind="walker", blocked="no"),
        case(True, kind="walker", blocked="no"),
        case(False, kind="walker", blocked="yes"),
    ]
    grown = (Clause((LEGAL, Literal("kind", (Constant("walker"),)).denied)),)

    excluding = learner.excluding(grown, examples, InferenceBudget(10.0))

    assert excluding[0][1]
    assert learner.allows(excluding, examples[0]) and not learner.allows(excluding, examples[2])


def test_a_clause_that_wrongly_covers_nothing_needs_no_exclusion() -> None:
    learner = ClauseLearner()
    examples = [case(True, kind="walker"), case(False, kind="flyer")]
    grown = learner.learn(examples, LEGAL, InferenceBudget(10.0))

    excluding = learner.excluding(grown, examples, InferenceBudget(10.0))

    assert excluding[0][1] == ()


def test_an_exclusion_is_learned_from_its_own_clause_s_cases_and_never_reaches_a_sibling() -> None:
    learner = ClauseLearner()
    examples = [
        case(True, kind="walker", blocked="no"),
        case(False, kind="walker", blocked="yes"),
        case(True, kind="flyer", blocked="yes"),
    ]
    grown = (
        Clause((LEGAL, Literal("kind", (Constant("walker"),)).denied)),
        Clause((LEGAL, Literal("kind", (Constant("flyer"),)).denied)),
    )

    excluding = learner.excluding(grown, examples, InferenceBudget(10.0))

    assert learner.allows(excluding, examples[2])


def test_two_clauses_split_by_side_are_merged_into_one_about_a_player_s_own() -> None:
    """Widening against cases cannot do this: once one clause covers what one side does and another covers the
    other, neither is ever short of a case to take in, so neither is ever widened again. Generalising the clauses
    against each other is what collapses them."""
    learner = ClauseLearner()
    white = Clause((LEGAL, Literal("turn", (Constant("first"),)).denied, Literal("owner", (Constant("first"),)).denied))
    black = Clause((LEGAL, Literal("turn", (Constant("second"),)).denied, Literal("owner", (Constant("second"),)).denied))
    examples = [
        case(True, turn="first", owner="first"),
        case(True, turn="second", owner="second"),
        case(False, turn="first", owner="second"),
        case(False, turn="second", owner="first"),
    ]

    merged = learner.merged((white, black), examples)

    assert len(merged) == 1
    assert all(learner.covers(merged[0], one) for one in examples if one.holds)
    assert not any(learner.covers(merged[0], one) for one in examples if not one.holds)


def test_clauses_that_cannot_merge_without_letting_things_through_are_left_apart() -> None:
    learner = ClauseLearner()
    one = Clause((LEGAL, Literal("kind", (Constant("walker"),)).denied))
    other = Clause((LEGAL, Literal("kind", (Constant("jumper"),)).denied))
    examples = [case(True, kind="walker"), case(True, kind="jumper"), case(False, kind="flyer")]

    assert len(learner.merged((one, other), examples)) == 2


def test_a_reading_of_one_thing_is_never_generalised_with_a_reading_of_another() -> None:
    """Opening up what a reading is *about* gives "the row of something is something", which holds of every case."""
    learner = ClauseLearner()
    clause = Clause((LEGAL, Literal("row", (Constant("target"), Constant(2))).denied))
    about_another = Example((Literal("row", (Constant("source"), Constant(5))),), True)

    assert learner.generalised(clause, about_another) is None


def test_a_reading_of_the_same_thing_is_generalised_on_what_was_read_of_it() -> None:
    learner = ClauseLearner()
    clause = Clause((LEGAL, Literal("row", (Constant("target"), Constant(2))).denied))
    about_the_same = Example((Literal("row", (Constant("target"), Constant(5))),), True)

    wider = learner.generalised(clause, about_the_same)

    assert wider is not None and wider.body[0].arguments[0] == Constant("target")


def test_a_reading_saying_nothing_at_all_is_dropped_rather_than_carried() -> None:
    learner = ClauseLearner()
    clause = Clause((
        LEGAL,
        Literal("kind", (Constant("walker"),)).denied,
        Literal("clock", (Constant(0),)).denied,
    ))

    wider = learner.generalised(clause, Example(
        (Literal("kind", (Constant("walker"),)), Literal("clock", (Constant(7),))), True
    ))

    assert wider is not None
    assert [one.predicate for one in wider.body] == ["kind"]
