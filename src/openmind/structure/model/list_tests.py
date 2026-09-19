from openmind.structure.model.list import List


def test_a_list_changes_into_a_new_list() -> None:
    hand = List(("king", "queen"))

    assert hand.appended("jack") == List(("king", "queen", "jack"))
    assert hand.removed(0) == List(("queen",)) and hand.replaced(1, "ace") == List(("king", "ace"))
    assert (hand[0], len(hand), "queen" in hand, hand.count("king")) == ("king", 2, True, 1)
    assert hand == List(("king", "queen"))
