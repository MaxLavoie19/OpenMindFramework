"""Putting heuristics beside a teller over the positions a game's decisions were made in."""

from openmind.training.model.decided import Decided
from openmind.training.service.teller_judging import TellerJudging


class APlayers:
    def __init__(self, names):
        self.names = names


class AGame:
    def __init__(self, names=("white", "black")):
        self._names = names

    def players(self):
        return APlayers(self._names)


class ANode:
    """A position that is only a number. Nothing here reads one — what a heuristic makes of it comes from the
    model, and what the teller makes of it comes from the teller."""

    def __init__(self, number):
        self.number = number
        self.game = AGame()


def a_decision(number, player="white"):
    return Decided(ANode(number), (), None, player, 1.0)


class AModel:
    """A heuristic as this sees one: something a valuer turns into a value per player."""

    def __init__(self, by):
        self.by = by


class AValuer:
    def __init__(self, model):
        self.model = model

    def values(self, model, node):
        found = model.by(node.number)
        return None if found is None else (found, -found)


def judging(**held):
    """A judging whose valuer is ours, so the measurement is tested and the rule engine is not."""
    one = TellerJudging(**held)
    one._valuing = lambda model: (
        lambda node, player: (lambda got: None if got is None else got[0 if player == "white" else 1])(
            AValuer(model).values(model, node)
        )
    )
    return one


def test_a_heuristic_ordering_positions_the_teller_s_way_tracks_it():
    decisions = [a_decision(one) for one in range(8)]

    found = judging().judged(
        decisions, [("with", AModel(lambda n: n))], lambda one: f"fen {one.node.number}",
        lambda names: [float(one.split()[-1]) for one in names],
    )

    assert found[0].holder == "with" and found[0].agreed > 0.99
    assert found[0].decided == 8


def test_a_position_the_teller_declines_is_dropped_from_both_lists_together():
    """A position it would not speak about cannot be a row. Dropped from the teller's list and not from the
    decisions', every heuristic after it is measured against the wrong board — which is a wrong number and no
    error anywhere."""
    decisions = [a_decision(one) for one in range(6)]

    found = judging().judged(
        decisions, [("with", AModel(lambda n: n))], lambda one: f"fen {one.node.number}",
        lambda names: [None if at == 2 else float(one.split()[-1]) for at, one in enumerate(names)],
    )

    assert found[0].agreed > 0.99, "the five it did speak about are still in the teller's order"
    assert found[0].decided == 5


def test_a_decision_that_cannot_be_named_is_never_put_to_the_teller():
    """Naming is the caller's and may fail — a position the domain cannot write down is not one to ask about,
    and a blank name asked anyway is a row the teller answers about nothing."""
    asked = []
    decisions = [a_decision(one) for one in range(4)]

    judging().judged(
        decisions, [("with", AModel(lambda n: n))],
        lambda one: "" if one.node.number == 1 else f"fen {one.node.number}",
        lambda names: asked.extend(names) or [float(one.split()[-1]) for one in names],
    )

    assert asked == ["fen 0", "fen 2", "fen 3"]


def test_too_few_positions_to_order_is_no_measurement_rather_than_a_score():
    decisions = [a_decision(one) for one in range(4)]

    found = judging().judged(
        decisions, [("with", AModel(lambda n: n))], lambda one: f"fen {one.node.number}",
        lambda names: [1.0] + [None] * (len(names) - 1),
    )

    assert found == ()


def test_every_heuristic_is_measured_over_the_same_positions_in_one_pass():
    """The teller is asked once for the lot, because starting it is what it costs — so the positions are
    gathered before any heuristic is asked anything."""
    times = []
    decisions = [a_decision(one) for one in range(8)]

    found = judging().judged(
        decisions,
        [("with", AModel(lambda n: n)), ("against", AModel(lambda n: -n)), ("silent", AModel(lambda n: None))],
        lambda one: f"fen {one.node.number}",
        lambda names: times.append(1) or [float(one.split()[-1]) for one in names],
    )

    assert times == [1], "the teller is asked once and not once per heuristic"
    assert {one.holder for one in found} == {"with", "against", "silent"}
    assert [one.agreed > 0.99 for one in found if one.holder == "with"] == [True]
    assert [one.agreed < -0.99 for one in found if one.holder == "against"] == [True]
