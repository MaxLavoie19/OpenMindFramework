from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.forward_chainer import ForwardChainer
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number, Variable

WIDE, NARROW = Constant("wide"), Constant("narrow")


def fact(predicate: str, *arguments: object, name: str = "", probability: float = 1.0) -> Clause:
    return Clause((Literal(predicate, tuple(arguments)),), probability, name)  # type: ignore[arg-type]


def rule(head: Literal, *body: Literal, name: str = "") -> Clause:
    return Clause((head, *(one.denied for one in body)), 1.0, name)


def concluded(clauses: tuple[Clause, ...], budget: InferenceBudget | None = None) -> set[str]:
    found = ForwardChainer().chain(clauses, budget or InferenceBudget(5.0))
    return {derivation.conclusion.readable for derivation in found}


def test_a_rule_and_a_fact_conclude_a_new_fact() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", Constant("first")),
    )

    assert "wins(first)" in concluded(clauses)


def test_a_conclusion_is_itself_reasoned_from_so_a_chain_runs_as_far_as_it_goes() -> None:
    clauses = (
        rule(Literal("worth at least", (Variable("T"), Variable("N"))), Literal("affords", (Variable("T"), Variable("N")))),
        rule(
            Literal("worth at least", (Variable("T"), Variable("N"))),
            Literal("takes in", (Variable("T"), Variable("O"))),
            Literal("worth at least", (Variable("O"), Variable("N"))),
        ),
        fact("affords", NARROW, Number(14)),
        fact("takes in", WIDE, NARROW),
    )

    found = concluded(clauses)

    assert "worth at least(narrow, 14)" in found
    assert "worth at least(wide, 14)" in found


def test_what_a_thing_is_worth_by_taking_in_another_rests_on_two_conclusions_and_no_rule_states_it() -> None:
    clauses = (
        rule(Literal("worth at least", (Variable("T"), Variable("N"))), Literal("affords", (Variable("T"), Variable("N")))),
        rule(
            Literal("worth at least", (Variable("T"), Variable("N"))),
            Literal("takes in", (Variable("T"), Variable("O"))),
            Literal("worth at least", (Variable("O"), Variable("N"))),
        ),
        fact("affords", NARROW, Number(14), name="what the narrow one affords"),
        fact("takes in", WIDE, NARROW, name="the wide one takes in the narrow one"),
    )

    found = ForwardChainer().chain(clauses, InferenceBudget(5.0))
    wide = next(one for one in found if one.conclusion.readable == "worth at least(wide, 14)")

    assert set(wide.rests_on) >= {"what the narrow one affords", "the wide one takes in the narrow one"}


def test_more_at_stake_than_can_be_answered_follows_only_once_a_player_acts_once() -> None:
    at_stake = fact("at stake", WIDE, Number(2))
    principle = rule(
        Literal("cannot all escape", (Variable("T"),)),
        Literal("at stake", (Variable("T"), Variable("N"))),
        Literal("more", (Variable("N"), Number(1))),
        Literal("acts once", ()),
    )

    without = concluded((principle, at_stake))
    with_it = concluded((principle, at_stake, fact("acts once")))

    assert "cannot all escape(wide)" not in without
    assert "cannot all escape(wide)" in with_it


def test_nothing_at_stake_beyond_one_concludes_nothing_even_where_a_player_acts_once() -> None:
    clauses = (
        rule(
            Literal("cannot all escape", (Variable("T"),)),
            Literal("at stake", (Variable("T"), Variable("N"))),
            Literal("more", (Variable("N"), Number(1))),
            Literal("acts once", ()),
        ),
        fact("at stake", NARROW, Number(1)),
        fact("acts once"),
    )

    assert "cannot all escape(narrow)" not in concluded(clauses)


def test_a_weaker_conclusion_is_not_kept_where_a_stronger_one_is_already_held() -> None:
    clauses = (
        rule(Literal("at_least", (Variable("T"), Variable("N"))), Literal("affords", (Variable("T"), Variable("N")))),
        fact("at_least", NARROW, Number(100)),
        fact("affords", NARROW, Number(14)),
    )

    found = concluded(clauses)

    assert "at_least(narrow, 100)" in found
    assert "at_least(narrow, 14)" not in found


def test_the_clauses_given_come_back_among_the_conclusions_resting_on_themselves() -> None:
    given = fact("acts once")

    found = ForwardChainer().chain((given,), InferenceBudget(5.0))

    assert any(one.conclusion == given and one.steps[0].rule == "given" for one in found)


def test_chaining_settles_once_nothing_further_follows() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", Constant("first")),
    )

    first = concluded(clauses)
    again = concluded(clauses)

    assert first == again


def test_a_budget_of_a_few_steps_stops_it_and_keeps_what_it_had() -> None:
    clauses = (
        rule(Literal("worth at least", (Variable("T"), Variable("N"))), Literal("affords", (Variable("T"), Variable("N")))),
        fact("affords", NARROW, Number(14)),
    )

    found = ForwardChainer().chain(clauses, InferenceBudget(5.0, steps=1))

    assert len(found) >= 2


def test_a_conclusion_leaning_on_a_doubtful_clause_says_which_one_it_leans_on() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", Constant("first"), name="what was seen", probability=0.8),
    )

    found = ForwardChainer().chain(clauses, InferenceBudget(5.0))
    wins = next(one for one in found if one.conclusion.readable.startswith("0.8::wins"))

    assert wins.chances == ("what was seen",)


def test_a_conclusion_carries_the_steps_that_reached_it_in_order() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", Constant("first")),
    )

    found = ForwardChainer().chain(clauses, InferenceBudget(5.0))
    wins = next(one for one in found if one.conclusion.readable == "wins(first)")

    assert [step.number for step in wins.steps] == list(range(1, len(wins.steps) + 1))
    assert wins.steps[-1].rule == "resolved"
