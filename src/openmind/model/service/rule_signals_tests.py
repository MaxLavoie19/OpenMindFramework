import numpy as np
import pytest

from openmind.inference.model.expression import Expression
from openmind.model.factory.model_factory import create_rule_signals
from openmind.model.model.rule_candidate import RuleCandidate
from openmind.model.service.rule_signals import (
    FIRES_OFTEN, SAYS_SOMETHING_NEW, WENT_WITH_WINNING,
    FiresOften, MovedTheFit, SaysSomethingNew, WentWithWinning, explains, gated, shown,
)

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


def pairwise(candidates):
    """Novelty the way it was worked out before: one `corrcoef` a pair, in a Python loop.

    Kept as the thing the fast one is measured against. It is the definition — each term against every term
    the fit leant on harder — and the rewrite is only allowed to change what that costs."""
    order = sorted(range(len(candidates)), key=lambda at: -abs(candidates[at].weight))
    found = [0.0] * len(candidates)
    for place, at in enumerate(order):
        mine = np.nan_to_num(candidates[at].readings, nan=0.0)
        if not len(mine) or float(mine.std()) == 0.0:
            continue
        likest = 0.0
        for before in order[:place]:
            theirs = np.nan_to_num(candidates[before].readings, nan=0.0)
            if float(theirs.std()) == 0.0:
                continue
            held = float(np.corrcoef(mine, theirs)[0, 1])
            if np.isfinite(held):
                likest = max(likest, abs(held))
        found[at] = candidates[at].influence() * gated(shown(candidates[at])) * (1.0 - likest)
    return found


@pytest.mark.parametrize("how_many", [1, 2, 7, 60, 300])
def test_novelty_read_by_blocks_is_what_reading_every_pair_gave(how_many):
    """**The whole claim of the rewrite: same answer, different cost.** Each term is still read against every
    term the fit weighted more heavily — that is what the signal means — and what changed is that the pairs
    are a matrix product rather than an interpreted loop.

    Run across the block size, so the seams inside a block and between blocks are both covered: 300 terms at a
    block of 512 is one block, and the sizes below it walk the edges where a block has one row or none."""
    rng = np.random.default_rng(7)
    many = [
        candidate(list(rng.normal(size=20)), weight=float(rng.normal())) for _ in range(how_many)
    ]

    assert SaysSomethingNew().rates(many) == pytest.approx(pairwise(many), abs=1e-9)


def test_novelty_is_the_same_across_a_block_boundary():
    """A term in the second block is read against every term in the first, and a rewrite that forgot to would
    say the second block was all new — which is exactly the failure a block introduces and nothing else
    would catch."""
    rng = np.random.default_rng(11)
    many = [candidate(list(rng.normal(size=12)), weight=float(rng.normal())) for _ in range(40)]
    small, large = SaysSomethingNew(), SaysSomethingNew()
    small.BLOCK, large.BLOCK = 8, 4096

    assert small.rates(many) == pytest.approx(large.rates(many), abs=1e-9)
    assert small.rates(many) == pytest.approx(pairwise(many), abs=1e-9)


def test_a_term_saying_the_same_of_every_row_is_not_new():
    """It correlates with nothing, itself included. Nought, the same as before, and not one — a flat column is
    not the most novel thing in the set."""
    rated = SaysSomethingNew().rates([counter(), candidate([3.0] * 20)])

    assert rated[1] == 0.0


def square(fires) -> RuleCandidate:
    """A rule reading one square's owner: one wherever that square is theirs, nought everywhere else.

    Binary, which is the whole difficulty. Its reading says the same thing on every row it fires on, so what
    it tells us is *where* it fires and not what it reads there."""
    readings = np.zeros(len(PAYOFFS))
    readings[fires] = 1.0
    return candidate(list(readings))


def test_a_square_that_fires_often_and_picks_out_nothing_is_wanted_by_nobody():
    """**What a store of 904 rated rules showed.** Of the 376 reading a single square's owner and nothing
    else, novelty rated 192 highest and coverage another 139 — nearly all of them between the two. Neither was
    malfunctioning: a single square is unlike every other term by construction, and it fires whenever that
    square is occupied. They were getting what they asked for, and what they asked for explained nothing.

    `MovedTheFit` is left alone: it rated none of those 376 highest, because it asks whether a term changes
    the fitted value, which is a question a useless square already fails."""
    useless = square(list(range(0, len(PAYOFFS), 2)))

    wanted = {signal.name: signal.rates([useless])[0] for signal in create_rule_signals()}

    assert wanted[SAYS_SOMETHING_NEW] == 0.0, "novel, and about nothing"
    assert wanted[FIRES_OFTEN] == 0.0, "common, and about nothing"


def test_a_square_that_picks_out_winners_is_still_wanted():
    """The gate refuses what explains nothing, not what is simple. The same shape of rule, firing as often,
    differing only in which rows it picks, is bought as before."""
    telling = square(list(range(8)))

    wanted = {signal.name: signal.rates([telling])[0] for signal in create_rule_signals()}

    assert wanted[SAYS_SOMETHING_NEW] > 0.5
    assert wanted[FIRES_OFTEN] > 0.0


def test_a_rule_is_still_never_marked_down_for_firing_rarely():
    """The standing rule, and the thing the gate nearly broke. A term that fired on one row has not been shown
    to explain nothing — nothing can be shown about it at all, since a correlation wants two points. Gating it
    at nought would mark down every detector in the game at once."""
    rare = detector(right=1)

    assert FiresOften().rates([rare])[0] > 0.0, "the gate is silent where it cannot be measured"
    assert SaysSomethingNew().rates([rare])[0] > 0.0


def test_the_gate_says_nothing_rather_than_nought_where_it_cannot_be_told():
    """Nought is a real answer — a term measured and found to explain nothing. One row is not that."""
    assert explains(detector(right=1)) is None
    assert explains(candidate([3.0] * 20)) is None, "a term saying the same of every row explains nothing of it"
    assert explains(counter()) is not None


def test_a_detector_is_asked_where_it_fires_rather_than_what_it_reads_there():
    """**The flaw that made the gate useless at first.** A rule reading one square is one wherever it speaks,
    so its readings have no variation on those rows at all and a correlation there is undefined — the gate
    fell silent on exactly the terms it was built to catch. What a detector says is which rows it picks out.

    **And how often it fired is part of that answer.** Two rows of twenty cannot correlate far however
    perfectly they are chosen, and that is right: a detector seen twice has shown less than one seen five
    hundred times, whatever each got right. The gate says what has been *shown*, and little is shown by two
    rows. Nothing is marked down for being rare — a rare term is still bought by whichever signal wants it.
    """
    rare = square(list(range(2)))
    common = square(list(range(10)))

    assert explains(rare) is not None and explains(common) is not None
    assert explains(common) > explains(rare), "both pick only winners, and more firings have shown more"
