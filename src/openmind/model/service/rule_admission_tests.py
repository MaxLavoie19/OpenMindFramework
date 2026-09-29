from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from openmind.inference.model.expression import Expression
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_rule_admission, create_rule_signals
from openmind.model.model.rule_candidate import RuleCandidate
from openmind.model.service.rule_admission import RuleAdmission
from openmind.model.service.rule_budget import RuleBudget
from openmind.model.service.rule_price import RulePrice

pytestmark = pytest.mark.log_level("INFO")

#: A pool big enough that a rule costs real bits. Out of two hundred readings a one-clause term costs 9.2 of
#: them, a two-clause term 16.8 and a twelve-clause term 70.2, which is what the allowances below are read
#: against.
VOCABULARY = 200


def base(tmp_path: Path) -> tuple[KnowledgeBase, str]:
    knowledge = create_knowledge_base("admissions", tmp_path)
    return knowledge, knowledge.ensure_context("chess").id


def candidate(readings: Sequence[float], weight: float = 1.0, clauses: int = 2) -> RuleCandidate:
    return RuleCandidate(
        Expression(template="VIEW", clauses=clauses, plies=0),
        rule=None,  # type: ignore[arg-type]
        weight=weight,
        readings=np.array(readings, dtype=float),
        payoffs=np.array([1.0] * 10 + [-1.0] * 10, dtype=float),
    )


class Wanting:
    """A signal that wants exactly what it is told to, so a test can say what was wanted."""

    def __init__(self, name: str, wants: Sequence[float]) -> None:
        self._name, self._wants = name, wants

    @property
    def name(self) -> str:
        return self._name

    def rates(self, candidates: Sequence[RuleCandidate]) -> Sequence[float]:
        return list(self._wants)[: len(candidates)]


def admission(allowance: float) -> RuleAdmission:
    return RuleAdmission(RuleBudget(allowance), RulePrice())


def test_one_voucher_is_enough_and_a_minority_signal_admits_alone(tmp_path: Path) -> None:
    """The whole reason weights were rejected. A rule needs somebody willing to stake a budget on it, not a
    majority — so a signal that is right about a narrow class of rules can admit them against the rest."""
    knowledge, context = base(tmp_path)
    candidates = [candidate([1.0] * 20), candidate([2.0] * 20)]
    signals = [Wanting("the majority", [1.0, 0.0]), Wanting("the odd one out", [0.0, 1.0])]

    admitted = admission(20.0).admitted(knowledge, context, candidates, signals, VOCABULARY)

    assert admitted == {0: ("the majority",), 1: ("the odd one out",)}


def test_a_signal_buys_what_it_wanted_most_first(tmp_path: Path) -> None:
    """A purse that runs out should run out on what the signal wanted least, which is the only thing that
    makes a rating worth taking."""
    knowledge, context = base(tmp_path)
    candidates = [candidate([1.0] * 20), candidate([2.0] * 20), candidate([3.0] * 20)]
    signals = [Wanting("a signal", [0.1, 0.9, 0.5])]

    # A two-clause term out of two hundred readings costs 16.8 bits, so this round affords exactly one.
    admitted = admission(18.0).admitted(knowledge, context, candidates, signals, VOCABULARY)

    assert 1 in admitted, "what it wanted most"
    assert 0 not in admitted, "what it wanted least went unbought"


def test_a_rule_it_cannot_afford_does_not_cost_it_the_ones_it_can(tmp_path: Path) -> None:
    """A long rule a signal has not saved for should not shut out the short ones it wanted next. Written to
    stop at the first refusal, a signal would spend a round on nothing whenever its favourite was dear."""
    knowledge, context = base(tmp_path)
    dear = candidate([1.0] * 20, clauses=12)
    cheap = candidate([2.0] * 20, clauses=1)
    signals = [Wanting("a signal", [1.0, 0.5])]

    # Seventy bits against nine, so this round affords only the short rule.
    admitted = admission(12.0).admitted(knowledge, context, [dear, cheap], signals, VOCABULARY)

    assert 0 not in admitted, "it could not afford the long one"
    assert 1 in admitted, "and still bought the short one it wanted next"


def test_a_longer_rule_is_rarer_rather_than_refused(tmp_path: Path) -> None:
    """"More clauses are more expensive, so they exist but are rarer." A signal that saves for a round buys
    the rule it could not buy on one round's income."""
    knowledge, context = base(tmp_path)
    long = [candidate([1.0] * 20, clauses=12)]
    signals = [Wanting("a patient signal", [1.0])]
    # Twelve clauses cost 70.2 bits, which is six rounds of this income and none of one.
    admit = admission(12.0)

    assert admit.admitted(knowledge, context, long, signals, VOCABULARY) == {}

    for _ in range(6):
        bought = admit.admitted(knowledge, context, long, signals, VOCABULARY)
        if bought:
            break

    assert bought == {0: ("a patient signal",)}, "saved up, it can be had"


def test_nothing_is_admitted_where_there_are_no_signals(tmp_path: Path) -> None:
    """An economy with nobody in it buys nothing. Letting everything through instead would make an empty
    roster look like a permissive one, which is the opposite of what it is."""
    knowledge, context = base(tmp_path)

    assert admission(99.0).admitted(knowledge, context, [candidate([1.0] * 20)], [], VOCABULARY) == {}


def test_a_signal_that_asks_for_nothing_buys_nothing(tmp_path: Path) -> None:
    """A rate of nought is not a cheap preference, it is a signal saying it does not want the rule. Buying it
    anyway because there was money left would make the budget the proposer."""
    knowledge, context = base(tmp_path)
    signals = [Wanting("an uninterested signal", [0.0])]

    assert admission(99.0).admitted(knowledge, context, [candidate([1.0] * 20)], signals, VOCABULARY) == {}


def test_a_rule_several_signals_bought_is_credited_to_each_in_shares(tmp_path: Path) -> None:
    """What a heuristic turned out to be worth is a fact about the whole of it, so what separates the signals
    is how much of it each backed — and one of two backers backed half that rule, not all of it."""
    shares = admission(1.0).shares({0: ("first", "second"), 1: ("first",)})

    assert shares == {"first": pytest.approx(0.75), "second": pytest.approx(0.25)}
    assert sum(shares.values()) == pytest.approx(1.0), "the whole worth is paid out and no more"


def test_a_bigger_allowance_buys_more_rules(tmp_path: Path) -> None:
    """The one knob, and it is the caller's. Nothing inside chooses a threshold; how much comes through is
    how much was budgeted."""
    knowledge, context = base(tmp_path)
    candidates = [candidate([float(at + one) for at in range(20)]) for one in range(8)]
    signals = create_rule_signals()

    mean = create_knowledge_base("mean", tmp_path / "mean")
    lavish = create_knowledge_base("lavish", tmp_path / "lavish")
    few = admission(16.0).admitted(mean, mean.ensure_context("chess").id, candidates, signals, VOCABULARY)
    many = admission(400.0).admitted(lavish, lavish.ensure_context("chess").id, candidates, signals, VOCABULARY)

    assert len(many) > len(few)


def test_what_a_run_spent_and_earned_outlives_it(tmp_path: Path) -> None:
    """The ledger lives in the knowledge base, so a restart resumes what the last run learned about which
    signals are worth listening to rather than starting the economy over."""
    knowledge, context = base(tmp_path)
    candidates = [candidate([1.0] * 20, clauses=12)]
    admission(6.0).admitted(knowledge, context, candidates, [Wanting("a saver", [1.0])], VOCABULARY)

    again = create_knowledge_base("admissions", tmp_path)
    held = RuleBudget().held(again, again.context_named("chess").id, "a saver")

    assert held == pytest.approx(6.0), "it saved rather than spent, and the saving survived"


def test_the_real_roster_admits_some_candidates_and_not_others(tmp_path: Path) -> None:
    """The gate has to be a gate. Admitting everything makes the economy decoration, and admitting nothing
    makes a ponder that settled something settle nothing."""
    knowledge, context = base(tmp_path)
    rows = 20
    rng = np.random.default_rng(7)
    candidates = [candidate(list(rng.normal(size=rows)), weight=1.0 / (one + 1)) for one in range(12)]

    admitted = create_rule_admission(allowance=40.0).admitted(
        knowledge, context, candidates, create_rule_signals(), VOCABULARY
    )

    assert 0 < len(admitted) < len(candidates)
