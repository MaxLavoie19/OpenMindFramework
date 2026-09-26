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
    """What the two cases disagree about is opened up; what the reading is *of* is not. Two readings are used so
    that the variable is shared between them and the generalised clause still says something — a clause whose one
    reading has become "the target has some row" says nothing at all, and is dropped rather than kept."""
    learner = ClauseLearner()
    clause = Clause((
        LEGAL,
        Literal("row", (Constant("target"), Constant(2))).denied,
        Literal("column", (Constant("target"), Constant(2))).denied,
    ))
    about_the_same = Example(
        (Literal("row", (Constant("target"), Constant(5))), Literal("column", (Constant("target"), Constant(5)))),
        True,
    )

    wider = learner.generalised(clause, about_the_same)

    assert wider is not None
    assert all(one.arguments[0] == Constant("target") for one in wider.body)
    assert wider.body[0].arguments[-1] == wider.body[1].arguments[-1]


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


def rule(*body: Literal) -> Clause:
    return Clause((LEGAL, *(one.denied for one in body)))


def reading(name: str, *values: object) -> Literal:
    return Literal(name, tuple(Constant(one) for one in values))


def test_two_rules_that_are_one_rule_are_made_one() -> None:
    """Two kinds of thing moving one step each are one rule about moving one step, and saying it once is cheaper
    than saying it twice."""
    examples = [case(True, kind="walker", step=1), case(True, kind="runner", step=1)]
    clauses = (
        rule(reading("kind", "walker"), reading("step", 1)),
        rule(reading("kind", "runner"), reading("step", 1)),
    )

    found = ClauseLearner().distilled(clauses, examples)

    assert len(found) == 1
    assert reading("step", 1) in found[0].body


def test_two_rules_are_not_made_one_where_what_they_accounted_for_stops_being_accounted_for() -> None:
    """The case merging always got wrong. A step and a double step share that something moves up its column, and
    a rule saying only that accounts for neither distance in particular — so the pair stands, however much two
    rules cost. What may not be given up is not up for sale."""
    examples = [
        case(True, kind="pawn", row=2, step=2),
        case(True, kind="pawn", row=5, step=1),
        case(False, kind="pawn", row=5, step=2),
    ]
    clauses = (
        rule(reading("kind", "pawn"), reading("row", 2), reading("step", 2)),
        rule(reading("kind", "pawn"), reading("step", 1)),
    )

    found = ClauseLearner().distilled(clauses, examples)

    assert len(found) == 2


def test_four_rules_that_are_one_rule_are_made_one_even_where_no_two_of_them_pay_on_their_own() -> None:
    """Four ways of moving that are one way of moving. Folded two at a time, the rule covering a pair can cost as
    much as the pair it replaced and the folding stops there; the rule covering all four is a quarter of the
    price. What has to be offered is the whole fold, not only the pairs."""
    kinds = ("walker", "runner", "rider", "flyer")
    examples = [case(True, kind=one, step=1) for one in kinds] + [case(False, kind="walker", step=9)]
    clauses = tuple(rule(reading("kind", one), reading("step", 1)) for one in kinds)

    found = ClauseLearner().distilled(clauses, examples)

    assert len(found) == 1
    assert reading("step", 1) in found[0].body
    assert all(ClauseLearner().covered(found, one) for one in examples if one.holds)


def test_constraints_learned_in_different_positions_are_said_as_the_one_constraint_they_are() -> None:
    """Each arrives carrying what its own position happened to read, so neither says what the other says and
    nothing can see they were one constraint. The readings that never varied go first, and then they are one."""
    examples = [
        case(False, kind="rider", blocked=True, turn="white", where="one"),
        case(True, kind="rider", blocked=False, turn="white", where="one"),
        case(False, kind="rider", blocked=True, turn="black", where="two"),
        case(True, kind="rider", blocked=False, turn="black", where="two"),
    ]
    generator = rule(reading("kind", "rider"))
    pairs = (
        (
            generator,
            (
                Clause((Literal("refused anyway", ()), reading("blocked", True).denied, reading("turn", "white").denied)),
                Clause((Literal("refused anyway", ()), reading("blocked", True).denied, reading("turn", "black").denied)),
            ),
        ),
    )

    found = ClauseLearner().allowing(pairs, examples)

    assert len(found[0][1]) == 1
    assert found[0][1][0].body == (reading("blocked", True),)


def test_what_the_pair_allows_is_unchanged_by_saying_it_more_simply() -> None:
    learner = ClauseLearner()
    examples = [
        case(False, kind="rider", blocked=True, turn="white", where="one"),
        case(True, kind="rider", blocked=False, turn="white", where="one"),
        case(False, kind="rider", blocked=True, turn="black", where="two"),
        case(True, kind="rider", blocked=False, turn="black", where="two"),
    ]
    generator = rule(reading("kind", "rider"))
    pairs = (
        (
            generator,
            (
                Clause((Literal("refused anyway", ()), reading("blocked", True).denied, reading("turn", "white").denied)),
                Clause((Literal("refused anyway", ()), reading("blocked", True).denied, reading("turn", "black").denied)),
            ),
        ),
    )

    found = learner.allowing(pairs, examples)

    assert [learner.allows(found, one) for one in examples] == [learner.allows(pairs, one) for one in examples]


def test_a_rule_with_nothing_refusing_it_is_left_as_it_is() -> None:
    examples = [case(True, kind="rider", blocked=False)]
    pairs = ((rule(reading("kind", "rider")), ()),)

    assert ClauseLearner().allowing(pairs, examples) == pairs


def test_a_condition_that_held_of_every_case_is_gone_because_the_rule_without_it_is_the_same_rule() -> None:
    """Castling rights that never varied are a condition true of every case. The rule without it covers exactly
    what the rule with it covered, and being shorter it wins. Of two rules that answer alike, the simplest."""
    learner = ClauseLearner()
    examples = [
        case(True, kind="walker", castling="KQkq", where="one"),
        case(False, kind="runner", castling="KQkq", where="two"),
    ]
    clauses = (rule(reading("kind", "walker"), reading("castling", "KQkq")),)

    found = learner.distilled(clauses, examples)

    assert found[0].body == (reading("kind", "walker"),)
    assert [learner.covered(found, one) for one in examples] == [True, False]


def test_a_condition_that_did_not_hold_of_every_case_stays() -> None:
    learner = ClauseLearner()
    examples = [
        case(True, kind="walker", castling="KQkq", where="one"),
        case(False, kind="walker", castling="Kkq", where="two"),
        case(False, kind="runner", castling="KQkq", where="three"),
    ]
    clauses = (rule(reading("kind", "walker"), reading("castling", "KQkq")),)

    found = learner.distilled(clauses, examples)

    assert len(found[0].body) == 2


def test_a_rule_is_never_grown_with_a_condition_that_held_of_every_case() -> None:
    """Castling rights that never varied are true of every case, so they cannot be part of why some moves are
    legal and others are not. A rule grown here never picks them up, so nothing has to take them out again."""
    examples = [
        case(True, kind="walker", step=1, castling="KQkq"),
        case(False, kind="walker", step=9, castling="KQkq"),
    ]

    found = learned(examples)

    assert found and all(reading("castling", "KQkq") not in one.body for one in found)


def test_the_cheapest_way_of_saying_the_rules_is_taken_and_not_the_first_found() -> None:
    """Dropping a condition two rules share is a saving, and it is also the last thing showing they were one
    rule — which was the larger saving. Taking the first improvement found loses it."""
    examples = [case(True, kind="walker", step=1), case(True, kind="runner", step=1)]
    clauses = (
        rule(reading("kind", "walker"), reading("step", 1)),
        rule(reading("kind", "runner"), reading("step", 1)),
    )

    found = ClauseLearner().distilled(clauses, examples)

    assert len(found) == 1 and found[0].body == (reading("step", 1),)


def test_a_rule_is_narrowed_with_another_condition_rather_than_thrown_away() -> None:
    """A constraint that refused a move the game allows is wrong as it stands, not worthless. Saying more makes
    it cover less, and what it must go on covering is what it may say it from."""
    learner = ClauseLearner()
    clause = rule(reading("kind", "rider"))
    keeping = [case(False, kind="rider", blocked=True), case(False, kind="rider", blocked=True)]
    without = [case(True, kind="rider", blocked=False)]

    found = learner.narrowed(clause, keeping, without)

    assert found is not None
    assert reading("kind", "rider") in found.body and reading("blocked", True) in found.body
    assert all(learner.covers(found, one) for one in keeping)
    assert not learner.covers(found, without[0])


def test_nothing_comes_back_where_no_reading_tells_the_two_apart() -> None:
    """Not a failure of the rule but of what could be read of the cases."""
    learner = ClauseLearner()
    clause = rule(reading("kind", "rider"))
    keeping = [case(False, kind="rider", blocked=True)]
    without = [case(True, kind="rider", blocked=True)]

    assert learner.narrowed(clause, keeping, without) is None
