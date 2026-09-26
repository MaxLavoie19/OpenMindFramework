from openmind.language.service.syntax_learner import SyntaxLearner

KINDS = ("H", "P")
COLUMNS = ("p", "q", "r")
ROWS = ("1", "2", "3")


def spoken():
    """A made-up game's notation, written every way it comes.

    Nothing here is chess and nothing says what any character is for. A kind is a capital, a column a lower-case
    letter and a row a digit — but only to us: all the learner is given is the strings."""
    said = []
    for column in COLUMNS:
        for row in ROWS:
            said.append(f"{column}{row}")
            for kind in KINDS:
                said.append(f"{kind}{column}{row}")
                said.append(f"{kind}x{column}{row}")
    return said * 5


def test_characters_standing_in_the_same_places_are_found_to_be_one_sort():
    """The columns are one sort and the rows another, because joining each collapses layouts and joining one
    across to the other does not."""
    grammar = SyntaxLearner().learn(spoken())

    assert grammar.sort_of("p") == grammar.sort_of("q") == grammar.sort_of("r")
    assert grammar.sort_of("1") == grammar.sort_of("2") == grammar.sort_of("3")
    assert grammar.sort_of("p") != grammar.sort_of("1")


def test_capitalisation_separates_without_anyone_saying_it_means_anything():
    """`P` names a kind and `p` a column. They are told apart by the company they keep, which is the only thing
    that could tell them apart in a game we know nothing about."""
    grammar = SyntaxLearner().learn(spoken())

    assert grammar.sort_of("P") != grammar.sort_of("p")
    assert grammar.sort_of("P") == grammar.sort_of("H")


def test_notations_of_one_layout_are_one_shape():
    """Which is what makes a place in a notation worth measuring: within a shape the layout is fixed."""
    grammar = SyntaxLearner().learn(spoken())

    assert grammar.shaped("p1") == grammar.shaped("q2")
    assert grammar.shaped("Hp1") == grammar.shaped("Pq2")
    assert grammar.shaped("p1") != grammar.shaped("Hp1")


def test_a_notation_we_have_no_shape_for_reads_as_nothing():
    """News rather than failure: the shapes found so far do not cover what the game just played."""
    grammar = SyntaxLearner().learn(spoken())

    assert grammar.shaped("z9") is None
    assert grammar.shaped("p1p1") is None


def test_nothing_said_is_a_grammar_with_nothing_in_it():
    grammar = SyntaxLearner().learn([])

    assert grammar.sorts == ()
    assert grammar.shapes == ()
