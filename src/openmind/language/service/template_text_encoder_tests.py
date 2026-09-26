from openmind.language.constant import meaning_constant as meaning
from openmind.language.service.template_text_encoder import TemplateTextEncoder


def said(node) -> str:
    return TemplateTextEncoder().encode(node)


def test_a_rule_reads_as_what_it_does_and_then_why():
    """A rule's meaning starts with its effect, because what a person wants first from an explanation is
    whether the thing is good or bad — the measure is the reason and comes after."""
    assert said({meaning.EFFECT: "raises", meaning.MEASURE: {meaning.NUMBER: 3}}) == "Raises my value: 3."
    assert said({meaning.EFFECT: "lowers", meaning.MEASURE: {meaning.NUMBER: 3}}) == "Lowers my value: 3."


def test_a_measure_on_its_own_reads_without_an_effect():
    """Half a heuristic is still worth reading: a term in a search, or a rule whose weight is not yet fitted,
    has a measure and no effect."""
    assert said({meaning.NUMBER: 7}) == "7"


def test_a_comparison_reads_as_a_sentence_rather_than_as_an_operator():
    """The point of the whole encoder: an explanation nobody can read explains nothing, and `>=` is not
    English."""
    node = {meaning.COMPARE: "at least", meaning.LEFT: {meaning.NUMBER: 2}, meaning.RIGHT: {meaning.NUMBER: 1}}

    assert said(node) == "2 is at least 1"


def test_a_comparison_against_a_player_reads_as_belonging_rather_than_as_equalling():
    """`the piece equals me` is not what the rule means. It means the piece is mine, and a reader told the
    first would have to work out the second every time."""
    node = {
        meaning.COMPARE: "equals",
        meaning.LEFT: {meaning.READING: "piece"},
        meaning.RIGHT: {meaning.PLAYER: "me"},
    }

    assert said(node).endswith("is mine")


def test_a_player_reads_as_who_they_are_to_the_reader():
    """A heuristic is written for whoever is being valued, so `me` is I and the other is the opponent — the
    same rule read for the other side reads the other way round."""
    assert said({meaning.PLAYER: "me"}) == "I"
    assert said({meaning.PLAYER: "opponent"}) == "the opponent"


def test_nothing_and_off_the_board_are_different_words():
    """An empty square holds nothing; outside is not a square at all. A reader given one word for both could
    not tell a piece on the edge from a piece beside a gap."""
    assert said({meaning.EMPTY: True}) == "empty"
    assert said({meaning.OUTSIDE: True}) == "outside"


def test_how_many_moves_somebody_has_reads_as_that():
    assert said({meaning.MOVES_AVAILABLE_TO: "me"}) == "the number of moves I could make"


def test_a_truth_reads_in_lower_case_as_english_writes_it():
    assert said({meaning.TRUTH: True}) == "true"


def test_a_node_nobody_wrote_a_template_for_is_shown_rather_than_dropped():
    """A reading nobody has words for is still something the rule said, and an explanation that silently
    omits part of itself is worse than one that reads awkwardly."""
    found = said({"something nobody named": 5})

    assert "something nobody named" in found
