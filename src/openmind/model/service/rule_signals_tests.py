import numpy as np
import pytest

from openmind.inference.model.expression import Expression
from openmind.model.factory.model_factory import create_rule_signals
from openmind.model.model.rule_candidate import RuleCandidate
from openmind.model.service.rule_signals import FiresOften, MovedTheFit, SaysSomethingNew, WentWithWinning

pytestmark = pytest.mark.log_level("INFO")

#: What a game paid over twenty positions, half won and half lost.
PAYOFFS = np.array([1.0] * 10 + [-1.0] * 10)


def candidate(readings: list[float], weight: float = 1.0, clauses: int = 2) -> RuleCandidate:
    return RuleCandidate(
        Expression(template="VIEW", clauses=clauses, plies=0),
        rule=None,  # type: ignore[arg-type]
        weight=weight,
        readings=np.array(readings, dtype=float),
        payoffs=PAYOFFS,
    )


def detector(right: int = 2) -> RuleCandidate:
    """A mate detector: nought on almost every position, and on the few where it speaks it is right."""
    return candidate([1.0] * right + [0.0] * (10 - right) + [0.0] * 10)


def counter() -> RuleCandidate:
    """A material count: it speaks everywhere and is a little right each time."""
    return candidate([0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2, 0.55, 0.45, 0.65] + [-0.5] * 10)


def test_a_term_is_read_over_the_rows_it_fired_on_and_not_over_all_of_them() -> None:
    """The mistake this project has a standing rule against, happening in a signal. A detector reads nought
    nearly everywhere, and 0 is not blank — it looked and found nothing. Scored over every row it would look
    like a term that says nothing; asked where it spoke, it is right every time."""
    rated = WentWithWinning().rates([detector(right=2)])

    assert rated[0] == pytest.approx(0.0), "two rows agreeing is nothing to read a correlation from"

    told = WentWithWinning().rates([candidate([1.0, 1.0, 2.0] + [0.0] * 7 + [-1.0, -2.0, -1.0] + [0.0] * 7)])

    assert told[0] > 0.9, "where it spoke, it went with winning"


def test_a_rare_sharp_term_is_wanted_as_much_as_a_common_one_that_is_right_as_often() -> None:
    """The whole of why coverage is never folded into a rating. A rule that fires on a twentieth of positions
    and is right every time is worth keeping; a signal that marked it down for rarity would destroy exactly
    those rules."""
    rare = candidate([3.0, 1.0] + [0.0] * 8 + [-3.0, -1.0] + [0.0] * 8)
    common = candidate([3.0, 1.0] * 5 + [-3.0, -1.0] * 5)
    rated = WentWithWinning().rates([rare, common])

    assert rated[0] == pytest.approx(rated[1], abs=0.05)


def test_what_the_fit_misses_most_is_what_it_leant_on_hardest() -> None:
    """Leave-one-out without refitting: a fitted value is the sum of its weighted readings, so the value
    without a term is the value less that term's weighted reading. What would be a refit per rule is a
    subtraction."""
    heavy = candidate([1.0] * 20, weight=2.0)
    light = candidate([1.0] * 20, weight=0.1)
    rated = MovedTheFit().rates([heavy, light])

    assert rated[0] > rated[1]


def test_a_term_that_says_what_a_stronger_one_already_said_is_not_new() -> None:
    """Redundancy read against a fixed order — the one the fit settled — so nothing depends on which signal
    was asked first. A rating that moves with the asking order is not a rating."""
    leading = candidate([float(at) for at in range(20)], weight=5.0)
    echo = candidate([float(at) * 2.0 for at in range(20)], weight=1.0)
    unlike = candidate([float(at % 3) for at in range(20)], weight=1.0)
    rated = SaysSomethingNew().rates([leading, echo, unlike])

    assert rated[0] > 0.0, "nothing stronger said it yet"
    assert rated[1] == pytest.approx(0.0, abs=1e-9), "the same term twice says nothing new"
    assert rated[2] > rated[1]


def test_the_coverage_signal_is_the_only_one_that_asks_for_the_generic_rule() -> None:
    """Why the fourth signal is there. Both kinds of rule are wanted, and the three sharpness signals all
    reward a term for being right where it fired — which is what a specific rule is good at. Without a signal
    asking for coverage, nothing in the economy buys the rule that holds an ordinary position together."""
    rated = FiresOften().rates([detector(right=1), counter()])

    assert rated[0] < rated[1]
    assert rated[0] > 0.0, "and it still sees the rare one; it simply wants it less"


def test_a_coverage_signal_declines_to_spend_and_never_marks_a_rule_down() -> None:
    """The distinction the standing rule turns on. Folding coverage into a score would take the rare rule's
    worth away from every signal at once. Here the rarity costs it nothing but one signal's interest, and a
    rule needs one voucher, not four."""
    rare = detector(right=1)
    wanted = [signal.rates([rare])[0] for signal in create_rule_signals()]

    assert any(one > 0.0 for one in wanted), "something still wants it"


def test_a_term_that_never_varies_is_asked_for_by_nobody() -> None:
    """A column that reads the same everywhere tells no two rows apart, so there is nothing in it to buy."""
    flat = candidate([1.0] * 20, weight=0.0)

    assert all(signal.rates([flat])[0] == 0.0 for signal in create_rule_signals() if signal.name != "fires often enough to know")


def test_no_rows_rates_at_nothing_rather_than_raising() -> None:
    """A ponder that settled nothing hands back empty columns, and a signal is not where that should surface."""
    empty = RuleCandidate(
        Expression(template="VIEW", clauses=1, plies=0),
        rule=None,  # type: ignore[arg-type]
        weight=1.0,
        readings=np.array([], dtype=float),
        payoffs=np.array([], dtype=float),
    )

    assert all(signal.rates([empty])[0] == 0.0 for signal in create_rule_signals())


def test_a_term_too_quiet_to_change_anything_is_wanted_by_nobody() -> None:
    """**The defect this fixes, measured in a real fitted heuristic.** Two terms came out at 0.494 and five
    more between 0.00072 and 0.000036 — fourteen thousand times too small to reorder anything the first two
    had separated — and three of the four signals bought them anyway, because they read only what a term says
    and never how loudly the fit lets it say so.

    Not a threshold: the quiet term is still wanted, just wanted so much less that a signal spending on what
    it wants most never reaches it."""
    loud = candidate([3.0, 1.0] * 5 + [-3.0, -1.0] * 5, weight=0.494)
    quiet = candidate([3.0, 1.0] * 5 + [-3.0, -1.0] * 5, weight=0.000036)

    for signal in create_rule_signals():
        rated = signal.rates([loud, quiet])
        assert rated[0] > rated[1], f"{signal.name} cannot hear how loudly a term may speak"
        assert rated[1] < rated[0] / 1000, f"{signal.name} wants the quiet one barely at all"
