from pathlib import Path

import pytest

from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_rule_budget
from openmind.model.service.rule_budget import ALLOWANCE

pytestmark = pytest.mark.log_level("INFO")


def base(tmp_path: Path) -> tuple[KnowledgeBase, str]:
    knowledge = create_knowledge_base("budgets", tmp_path)
    return knowledge, knowledge.ensure_context("chess").id


def test_a_signal_starts_with_nothing_and_is_given_an_income(tmp_path: Path) -> None:
    """A signal holds nothing until a round pays it. What keeps it from being shut out is the income, not a
    floor under what it holds — a floor under holdings is an infinite well, since anything priced at or under
    it could be bought for ever however wrong the signal had been."""
    knowledge, context = base(tmp_path)
    budget = create_rule_budget()

    assert budget.held(knowledge, context, "counts what is held") == 0.0
    budget.allowed(knowledge, context, ("counts what is held",))
    assert budget.held(knowledge, context, "counts what is held") == ALLOWANCE


def test_vouching_spends_and_a_signal_cannot_spend_what_it_has_not_got(tmp_path: Path) -> None:
    knowledge, context = base(tmp_path)
    budget = create_rule_budget()
    budget.earned(knowledge, context, {"a signal": 1.0}, worth=4.0)

    assert budget.vouched(knowledge, context, "a signal", price=3.0)
    assert budget.held(knowledge, context, "a signal") == pytest.approx(1.0)
    assert not budget.vouched(knowledge, context, "a signal", price=99.0)
    assert budget.held(knowledge, context, "a signal") == pytest.approx(1.0), "a refused vouch costs nothing"


def test_a_signal_is_paid_for_the_share_of_the_ruleset_it_vouched_for(tmp_path: Path) -> None:
    """The worth is the whole ruleset's, because what a heuristic is worth is a fact about the whole of it.
    What separates the signals is how much of it each one backed."""
    knowledge, context = base(tmp_path)
    budget = create_rule_budget()

    budget.earned(knowledge, context, {"most of it": 0.8, "a little of it": 0.2}, worth=10.0)

    assert budget.held(knowledge, context, "most of it") == pytest.approx(8.0)
    assert budget.held(knowledge, context, "a little of it") == pytest.approx(2.0)


def test_a_bad_vouch_decays_the_budget_and_never_empties_it(tmp_path: Path) -> None:
    """A signal bankrupted by a run of bad luck could never buy again, so could never gather the evidence
    that would show the luck for what it was — which is the tyranny weights were rejected for, by another
    road. Decay compounds, so being wrong repeatedly still costs."""
    knowledge, context = base(tmp_path)
    budget = create_rule_budget()
    budget.earned(knowledge, context, {"a wrong signal": 1.0}, worth=20.0)

    held = [budget.held(knowledge, context, "a wrong signal")]
    for _ in range(40):
        budget.earned(knowledge, context, {"a wrong signal": 1.0}, worth=-0.5)
        held.append(budget.held(knowledge, context, "a wrong signal"))

    assert held == sorted(held, reverse=True), "being wrong costs every time"
    assert held[-1] < held[0] / 10, "and it compounds"
    assert held[-1] >= ALLOWANCE, "but decay alone never takes away the next round's try"
    assert budget.afford(knowledge, context, "a wrong signal", price=ALLOWANCE)


def test_a_signal_that_earns_outbuys_one_that_does_not(tmp_path: Path) -> None:
    """The whole point of a budget: proportion rather than a winner. The poorer signal still buys, just less
    often, which is what a weight could not give."""
    knowledge, context = base(tmp_path)
    budget = create_rule_budget()
    for _ in range(10):
        budget.earned(knowledge, context, {"good": 1.0}, worth=1.0)
        budget.earned(knowledge, context, {"poor": 1.0}, worth=0.25)

    bought = {}
    for signal in ("good", "poor"):
        count = 0
        while budget.vouched(knowledge, context, signal, price=1.0) and count < 1000:
            count += 1
        bought[signal] = count

    assert bought["good"] > bought["poor"]
    assert bought["poor"] > 0, "a signal that earns little still buys something"


def test_what_a_run_learns_about_its_signals_outlives_it(tmp_path: Path) -> None:
    knowledge, context = base(tmp_path)
    create_rule_budget().earned(knowledge, context, {"a signal": 1.0}, worth=6.0)

    again = create_knowledge_base("budgets", tmp_path)
    budget = create_rule_budget()

    assert budget.held(again, again.context_named("chess").id, "a signal") == pytest.approx(6.0)


def test_a_signal_that_has_spent_everything_is_broke_until_the_next_round(tmp_path: Path) -> None:
    """Broke and not shut out. Spending can empty a purse — that is what makes a budget a budget — and the
    round after, the income lets it try again."""
    knowledge, context = base(tmp_path)
    budget = create_rule_budget()
    budget.allowed(knowledge, context, ("a signal",))

    assert budget.vouched(knowledge, context, "a signal", price=ALLOWANCE)
    assert budget.held(knowledge, context, "a signal") == 0.0
    assert not budget.vouched(knowledge, context, "a signal", price=ALLOWANCE)

    budget.allowed(knowledge, context, ("a signal",))

    assert budget.vouched(knowledge, context, "a signal", price=ALLOWANCE)
