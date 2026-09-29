import random
from pathlib import Path

import pytest

from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_factor_weights

pytestmark = pytest.mark.log_level("INFO")

PRICE, PRICES = "price", ("0.0", "0.001", "0.01")


def base(tmp_path: Path) -> tuple[KnowledgeBase, str]:
    knowledge = create_knowledge_base("factors", tmp_path)
    return knowledge, knowledge.ensure_context("chess").id


def test_a_way_of_making_that_has_never_been_tried_is_drawn_first(tmp_path: Path) -> None:
    """A value never tried has everything to prove, so it goes before one that has paid well. That is what
    keeps a pool of ways being explored rather than settling on whichever was tried first."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()
    weights.paid(knowledge, context, {PRICE: "0.0"}, worth=0.9)

    drawn = {weights.drawn(knowledge, context, PRICE, PRICES, random.Random(seed)) for seed in range(20)}

    assert "0.0" not in drawn, "what has been measured waits while anything is unmeasured"
    assert drawn == {"0.001", "0.01"}


def test_the_way_that_has_paid_is_drawn_more_often_once_all_have_been_tried(tmp_path: Path) -> None:
    """The whole point: the factor value coherent with good heuristics gets more of the making."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()
    for _ in range(20):
        weights.paid(knowledge, context, {PRICE: "0.01"}, worth=0.8)
        weights.paid(knowledge, context, {PRICE: "0.001"}, worth=0.1)
        weights.paid(knowledge, context, {PRICE: "0.0"}, worth=0.1)

    counted: dict[str, int] = {}
    for seed in range(300):
        one = weights.drawn(knowledge, context, PRICE, PRICES, random.Random(seed))
        counted[one] = counted.get(one, 0) + 1

    assert counted["0.01"] > counted["0.001"]
    assert counted["0.01"] > counted["0.0"]
    assert min(counted.values()) > 0, "the rest are still tried, which is what lets a bad start be undone"


def test_a_way_worth_less_than_nothing_can_still_be_drawn_again(tmp_path: Path) -> None:
    """A heuristic can be worth less than nothing — it expected the losing move more often than ignorance
    would have. A share of a negative number is not a share, so the worst is moved to nought rather than made
    impossible: the first unlucky measurement must not be the last word."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()
    for _ in range(10):
        weights.paid(knowledge, context, {PRICE: "0.0"}, worth=-0.5)
        weights.paid(knowledge, context, {PRICE: "0.001"}, worth=0.4)
        weights.paid(knowledge, context, {PRICE: "0.01"}, worth=0.4)

    drawn = {weights.drawn(knowledge, context, PRICE, PRICES, random.Random(seed)) for seed in range(200)}

    assert "0.0" in drawn


def test_what_a_way_of_making_is_worth_is_the_mean_of_what_it_made(tmp_path: Path) -> None:
    """What is wanted is what that way is worth, not what the last heuristic made with it happened to be."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()

    for worth in (1.0, 0.0, 1.0, 0.0):
        weights.paid(knowledge, context, {PRICE: "0.01"}, worth=worth)

    belief = knowledge.belief("worth of price being 0.01", context)
    assert belief.value == pytest.approx(0.5)
    assert dict(belief.tags)["made"] == 4


def test_every_factor_that_made_a_heuristic_takes_the_whole_of_what_it_was_worth(tmp_path: Path) -> None:
    """Splitting the credit would need to know how much of the heuristic each choice accounted for, which is
    the question the drawing exists because nobody can answer. Drawn independently, the factors tell
    themselves apart over many heuristics instead."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()

    weights.paid(knowledge, context, {PRICE: "0.01", "admission": "every term the fit kept"}, worth=0.7)

    assert knowledge.belief("worth of price being 0.01", context).value == pytest.approx(0.7)
    assert knowledge.belief("worth of admission being every term the fit kept", context).value == pytest.approx(0.7)


def test_a_factor_with_one_value_is_no_choice_and_a_factor_with_none_says_so(tmp_path: Path) -> None:
    """A caller offering one way of making is saying there is nothing to decide here, which is not an error;
    offering none has asked for something that cannot be answered."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()

    assert weights.drawn(knowledge, context, PRICE, ("0.01",), random.Random(1)) == "0.01"
    with pytest.raises(ValueError, match="values"):
        weights.drawn(knowledge, context, PRICE, (), random.Random(1))


def test_what_was_learned_outlives_the_run_that_learned_it(tmp_path: Path) -> None:
    """The weights are beliefs like anything else, so a restart goes on from what the last run found out
    about how to make a heuristic rather than starting the search again."""
    knowledge, context = base(tmp_path)
    weights = create_factor_weights()
    for _ in range(5):
        weights.paid(knowledge, context, {PRICE: "0.01"}, worth=0.8)

    again = create_knowledge_base("factors", tmp_path)
    belief = again.belief("worth of price being 0.01", again.context_named("chess").id)

    assert belief is not None
    assert belief.value == pytest.approx(0.8)
    assert dict(belief.tags)["made"] == 5
