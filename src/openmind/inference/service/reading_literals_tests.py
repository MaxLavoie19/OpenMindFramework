from openmind.inference.service.reading_literals import ReadingLiterals
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant


def only(readings: dict[str, object]) -> Literal:
    found = ReadingLiterals().literals(readings)  # type: ignore[arg-type]
    assert len(found) == 1
    return found[0]


def test_a_reading_of_one_place_becomes_a_literal_of_that_place_and_what_was_read() -> None:
    found = only({"piece at source": "walker"})

    assert found == Literal("at", (Constant("piece"), Constant("source"), Constant("walker")))


def test_a_reading_of_two_places_keeps_both_of_them_apart() -> None:
    found = only({"rows from source to target": 3})

    assert found == Literal("rows", (Constant("source"), Constant("target"), Constant(3)))


def test_the_same_question_asked_of_different_places_is_one_predicate_and_not_two() -> None:
    found = ReadingLiterals().literals({"rows from source to target": 1, "rows from target to source": -1})  # type: ignore[arg-type]

    assert {one.predicate for one in found} == {"rows"}


def test_a_reading_about_a_player_keeps_the_player_as_a_term_so_a_rule_can_quantify_over_it() -> None:
    found = only({"first can reach target": True})

    assert found.arguments[0] == Constant("first")


def test_readings_named_the_same_shape_still_come_apart_into_their_own_terms() -> None:
    """Three of the reach templates are the same shape once filled — `"{player} can reach {parameter}"`,
    `"{player} can reach {holding}"` and `"{player} can reach {whose} {holding}"` — so which one a filled name came
    from cannot be recovered, and it is not guessed at. What matters survives anyway: the player is a term of its
    own in each, so a clause can still quantify over it, which is the thing a flattened name made impossible."""
    found = ReadingLiterals().literals({"first can reach target": True, "second can reach piece 'walker'": 2})  # type: ignore[arg-type]

    assert all(one.arguments[0] in (Constant("first"), Constant("second")) for one in found)
    assert all(len(one.arguments) > 1 for one in found)


def test_whose_a_thing_is_keeps_the_side_apart_from_the_place() -> None:
    found = only({"piece at source is the one acting's": True})

    assert found.predicate == "belongs"
    assert found.arguments[:3] == (Constant("piece"), Constant("source"), Constant("the one acting's"))


def test_a_reading_of_the_position_the_action_leads_to_says_so_and_keeps_its_own_shape() -> None:
    found = only({"after it, piece at target": "walker"})

    assert found.predicate == "after it, at"
    assert found.arguments == (Constant("piece"), Constant("target"), Constant("walker"))


def test_a_reading_no_template_accounts_for_is_kept_whole_rather_than_thrown_away() -> None:
    found = only({"something a game named its own way": 4})

    assert found == Literal("something a game named its own way", (Constant(4),))


def test_a_whole_number_in_a_name_is_read_back_as_a_number() -> None:
    found = only({"row of source": 2})

    assert found.arguments == (Constant("source"), Constant(2))


def test_a_reading_whose_value_was_quoted_into_its_name_is_read_back_unquoted() -> None:
    found = only({"first can reach piece 'walker'": True})

    assert Constant("walker") in found.arguments


def test_readings_gathered_with_their_labels_become_cases_to_learn_from() -> None:
    seen = [({"piece at source": "walker"}, True), ({"piece at source": "flyer"}, False)]

    found = ReadingLiterals().examples(seen, ("a position", "another position"))  # type: ignore[arg-type]

    assert [one.holds for one in found] == [True, False]
    assert [one.where for one in found] == ["a position", "another position"]


def test_every_reading_of_a_case_becomes_a_literal_of_that_case() -> None:
    example = ReadingLiterals().example({"piece at source": "walker", "steps from source to target": 1}, True)  # type: ignore[arg-type]

    assert len(example.literals) == 2
    assert {one.predicate for one in example.literals} == {"at", "steps"}
