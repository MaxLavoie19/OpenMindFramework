from openmind.heuristic.service.cascade_order import CascadeOrder


def _scores(held: dict[str, list[float | None]]):
    """A way of scoring built from a table of what each rater made of each decision."""
    return lambda name, at: held[name][at]


def test_the_better_rater_goes_first_whatever_it_is() -> None:
    """Being specialised earns a rater nothing. If a general rater predicts this player's endings better
    than a rule set fitted on their endings, the general one takes the slot."""
    held = {"fitted on their endings": [0.1] * 12, "general": [0.9] * 12}

    order = CascadeOrder().ordered(("fitted on their endings", "general"), _scores(held), 12)

    assert order[0] == "general"


def test_raters_are_compared_only_where_all_of_them_answered() -> None:
    """Otherwise a rater that declines whenever a position is hard wins by dodging, and a comparison over
    different sets of decisions is not a comparison at all."""
    # The dodger scores brilliantly, but only on the decisions it did not decline — and those are the easy
    # ones, where the honest rater also scores well.
    held = {
        "the dodger": [None, None, 1.0, 1.0],
        "the honest one": [0.2, 0.2, 0.9, 0.9],
    }

    order = CascadeOrder().ordered(("the dodger", "the honest one"), _scores(held), 4)

    assert order[0] == "the dodger", "on the two it answered it really is better"
    # And the pairing is what makes that claim narrow rather than false: it won on two decisions, not four.


def test_a_margin_inside_the_noise_leaves_the_order_alone(caplog) -> None:
    """Churning on differences the decisions cannot resolve is how a system appears to improve while doing
    nothing."""
    import logging

    held = {"the incumbent": [0.5, 0.1, 0.9, 0.4], "the challenger": [0.51, 0.09, 0.92, 0.38]}

    with caplog.at_level(logging.INFO):
        CascadeOrder().ordered(("the incumbent", "the challenger"), _scores(held), 4)

    assert any("does not beat" in one.message for one in caplog.records), "it says the margin was not convincing"


def test_nothing_measurable_leaves_the_order_as_it_came() -> None:
    """Not knowing which of two is better is a reason to leave them alone, never a reason to rank them."""
    held = {"one": [None, None], "other": [0.9, 0.9]}

    assert CascadeOrder().ordered(("one", "other"), _scores(held), 2) == ("one", "other")


def test_one_rater_is_already_in_order() -> None:
    assert CascadeOrder().ordered(("alone",), _scores({"alone": [1.0]}), 1) == ("alone",)


def test_the_currency_is_the_caller_s() -> None:
    """A rater that plays well is judged on games won and one that says what somebody will do on how little
    it was surprised. Higher is better, and what higher means is not this one's business."""
    order = CascadeOrder()
    winning = {"plays well": [1.0] * 8, "plays badly": [0.0] * 8}
    surprised = {"plays well": [-2.0] * 8, "plays badly": [-0.1] * 8}

    assert order.ordered(("plays well", "plays badly"), _scores(winning), 8)[0] == "plays well"
    assert order.ordered(("plays well", "plays badly"), _scores(surprised), 8)[0] == "plays badly"
