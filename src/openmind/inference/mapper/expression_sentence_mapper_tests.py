import pytest

from openmind.inference.mapper.expression_sentence_mapper import ExpressionSentenceMapper


@pytest.mark.parametrize(
    ("source", "sentence"),
    [
        (
            "here.best(other, lambda v2: v2.worst(me, lambda v1: (sum(1 for at in v1.color if v1.color[at] == other)) "
            "- (sum(1 for at in v2.color if v2.color[at] == other))))",
            "the highest, over the opponent's moves, of the lowest, over my moves, of the change in the number of places "
            "where the color there is the opponent's",
        ),
        (
            "sum(1 for at in here.piece if here.piece[at] == 'knight' and here.color[at] == me)",
            "the number of places where the piece there is knight and the color there is mine",
        ),
        ("(here.mobility(me)) >= 30", "the number of moves I could make is at least 30"),
        (
            "sum(1 for at in here.cell if here.cell[at] == me and here.offset('cell', at, 0, 1) == me "
            "and here.offset('cell', at, 0, -1) == '<outside>')",
            "the number of places where the cell there is mine and the cell 0, 1 away from there is mine and the cell "
            "0, -1 away from there is outside",
        ),
        (
            "min(((abs((here.x[i]) - (here.x[j]))) for i in here.x for j in here.x if i != j), default=0)",
            "the lowest, over every pair of different entries, of the distance between the x of one entry and the x of "
            "the other entry",
        ),
        (
            "sum(1 for i in here.x for j in here.x if i != j if (here.x[i]) >= (here.x[j]))",
            "the number of pairs of different entries where the x of one entry is at least the x of the other entry",
        ),
        (
            "here.count(me, lambda v1: (v1.mobility(other)) < (here.mobility(other)))",
            "the number of my moves after which the number of moves the opponent could make is less than the number of "
            "moves the opponent could make now",
        ),
        (
            "here.worst(other, lambda v2: v2.best(me, lambda v1: v1.payoff[me] == 1.0))",
            "the lowest, over the opponent's moves, of the highest, over my moves, of whether my payoff is 1",
        ),
        (
            "here.best(other, lambda v2: v2.best(me, lambda v1: (v1.x[me]) - (v2.x[me] + here.x[other])))",
            "the highest, over the opponent's moves, of the highest, over my moves, of my x minus (my x after the "
            "opponent's move plus the opponent's x now)",
        ),
        ("(here.cell[2, 2] == me) * (here.turn == 'X')", "the cell at 2, 2 is mine times the turn is X"),
        (
            "(sum((here.x[i]) for i in here.x)) / max(1, here.x[other])",
            "the total, over every entry, of the x of that entry divided by the opponent's x, counted as at least 1",
        ),
        ("max(here.x[me], abs(here.x[other]))", "the larger of my x and the size of the opponent's x"),
        ("not (here.payoff['X'] == None)", "not (the payoff at X is empty)"),
        ("wins(other) >= 1", "how many winning moves the opponent would have is at least 1"),
        ("here.halfmove", "the halfmove"),
    ],
)
def test_expressions_read_construct_by_construct(source: str, sentence: str) -> None:
    assert ExpressionSentenceMapper().to_sentence(source) == sentence


def test_what_has_no_template_is_quoted_as_its_source() -> None:
    mapper = ExpressionSentenceMapper()

    assert mapper.to_sentence("lambda: 3") == "`lambda: 3`"
    assert mapper.to_sentence("here.x[me] ** 2 >= 1") == "`here.x[me] ** 2` is at least 1"
    assert mapper.to_sentence("here.x[") == "`here.x[`"
