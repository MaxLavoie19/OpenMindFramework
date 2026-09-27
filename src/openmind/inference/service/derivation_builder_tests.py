from openmind.inference.model.derivation_step import GIVEN
from openmind.inference.model.substitution import Substitution
from openmind.inference.service.derivation_builder import DerivationBuilder
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal


def said(name: str, *, certainty: float = 1.0, called: str = "") -> Clause:
    return Clause((Literal(name, ()),), probability=certainty, name=called)


def test_a_clause_taken_as_it_stands_rests_on_itself():
    made = DerivationBuilder().given(said("it rains"))

    assert made.conclusion == said("it rains")
    assert [(one.number, one.rule, one.premises) for one in made.steps] == [(1, GIVEN, ())]


def test_a_conclusion_records_the_step_that_reached_it_and_what_it_came_from():
    builder = DerivationBuilder()
    one, other = builder.given(said("it rains")), builder.given(said("the ground gets wet"))

    joined = builder.joined(said("the ground is wet"), "resolution", Substitution(), one, other)

    assert joined.conclusion == said("the ground is wet")
    last = joined.steps[-1]
    assert last.rule == "resolution"
    assert last.premises == (1, 2), "the last step of each parent, renumbered"


def test_each_parent_s_steps_are_renumbered_rather_than_laid_end_to_end_as_they_were():
    """The fiddly part it exists to get right. Each parent numbers its own steps from one, so laying two of them
    end to end would leave several steps called 3 and premises pointing at whichever came first."""
    builder = DerivationBuilder()
    left = builder.joined(said("b"), "a rule", Substitution(), builder.given(said("a")))
    right = builder.joined(said("d"), "a rule", Substitution(), builder.given(said("c")))

    joined = builder.joined(said("e"), "resolution", Substitution(), left, right)

    assert [one.number for one in joined.steps] == [1, 2, 3, 4, 5]
    assert len({one.number for one in joined.steps}) == len(joined.steps), "no two steps share a number"


def test_a_premise_points_at_the_step_it_now_is_rather_than_the_one_it_was():
    builder = DerivationBuilder()
    left = builder.joined(said("b"), "a rule", Substitution(), builder.given(said("a")))
    right = builder.joined(said("d"), "a rule", Substitution(), builder.given(said("c")))

    joined = builder.joined(said("e"), "resolution", Substitution(), left, right)

    by_number = {one.number: one for one in joined.steps}
    for step in joined.steps:
        for premise in step.premises:
            assert premise in by_number, "every premise names a step that is there"
            assert premise < step.number, "and one that came before it"
    assert by_number[4].premises == (3,), "the right-hand parent's premise moved with it"


def test_a_step_both_parents_share_is_kept_once():
    """A thing used twice was still only established once."""
    builder = DerivationBuilder()
    shared = builder.given(said("a"))
    left = builder.joined(said("b"), "a rule", Substitution(), shared)
    right = builder.joined(said("c"), "a rule", Substitution(), shared)

    joined = builder.joined(said("d"), "resolution", Substitution(), left, right)

    given = [one for one in joined.steps if one.rule == GIVEN and one.clause == said("a")]
    assert len(given) == 1
    assert [one.number for one in joined.steps] == [1, 2, 3, 4]


def test_a_doubtful_clause_a_conclusion_leans_on_is_named():
    made = DerivationBuilder().given(said("it rains", certainty=0.7, called="the forecast"))

    assert made.chances == ("the forecast",)


def test_a_certain_clause_is_not_a_chance():
    assert DerivationBuilder().given(said("it rains", called="the calendar")).chances == ()


def test_a_doubtful_clause_with_no_name_cannot_be_pointed_at_and_so_is_not_named():
    assert DerivationBuilder().given(said("it rains", certainty=0.7)).chances == ()


def test_leaning_twice_on_the_same_doubtful_clause_is_leaning_on_it_once():
    """What matters later is which ones a conclusion leans on. Two reasons for the same conclusion are two
    reasons only where they lean on different ones."""
    builder = DerivationBuilder()
    doubtful = said("it rains", certainty=0.7, called="the forecast")
    left = builder.given(doubtful)
    right = builder.given(doubtful)

    joined = builder.joined(said("wet"), "resolution", Substitution(), left, right)

    assert joined.chances == ("the forecast",)


def test_what_a_conclusion_leans_on_is_gathered_from_every_parent():
    builder = DerivationBuilder()
    left = builder.given(said("a", certainty=0.7, called="the forecast"))
    right = builder.given(said("b", certainty=0.5, called="the rumour"))

    joined = builder.joined(said("c"), "resolution", Substitution(), left, right)

    assert set(joined.chances) == {"the forecast", "the rumour"}


def test_a_conclusion_that_is_itself_doubtful_is_named_among_its_own_chances():
    builder = DerivationBuilder()
    parent = builder.given(said("a"))

    joined = builder.joined(said("b", certainty=0.5, called="the guess"), "a rule", Substitution(), parent)

    assert joined.chances == ("the guess",)


def test_a_conclusion_out_of_nothing_is_still_one_step():
    joined = DerivationBuilder().joined(said("a"), "assumed", Substitution())

    assert [(one.number, one.rule, one.premises) for one in joined.steps] == [(1, "assumed", ())]
