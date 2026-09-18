from openmind.doxastic.model.claim import Claim
from openmind.rbs.model.python_rule import PythonRule


def test_the_name_the_holder_and_the_subjects_identify_a_claim_whether_or_not_it_carries_its_rule() -> None:
    named = Claim("black is short of moves", about=("black",))
    with_rule = Claim("black is short of moves", PythonRule("here.mobility('black') < 5"), about=("black",))

    assert named.key == with_rule.key
    assert named != with_rule


def test_a_claim_about_another_player_s_belief_is_a_claim_of_its_own() -> None:
    mine = Claim("the knight is trapped")
    theirs = Claim("the knight is trapped", holder=("black",))
    deeper = Claim("the knight is trapped", holder=("black", "white"))

    assert len({mine.key, theirs.key, deeper.key}) == 3
    assert (mine.believed_by, theirs.believed_by, deeper.believed_by) == (None, "black", "white")


def test_the_subjects_identify_a_claim_whatever_order_they_are_given_in() -> None:
    assert Claim("they are trading", about=("white", "black")).key == Claim("they are trading", about=("black", "white")).key
