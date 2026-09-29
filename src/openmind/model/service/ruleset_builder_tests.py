import random

import numpy as np
import pytest

from openmind.heuristic.model.rule_cost import RuleCost
from openmind.inference.model.expression import Expression
from openmind.model.factory.model_factory import create_rule_signals, create_ruleset_builder
from openmind.model.model.rule_candidate import RuleCandidate
from openmind.rule.model.python_rule import PythonRule

pytestmark = pytest.mark.log_level("INFO")

PAYOFFS = np.array([1.0] * 10 + [-1.0] * 10)


def candidate(source: str, readings: list[float], weight: float = 1.0) -> RuleCandidate:
    return RuleCandidate(
        Expression(template="VIEW", clauses=2, plies=0),
        rule=PythonRule(source),
        weight=weight,
        readings=np.array(readings, dtype=float),
        payoffs=PAYOFFS,
    )


def sharp(source: str) -> RuleCandidate:
    """Narrow and accurate: it speaks on a fifth of positions and is right when it does."""
    return candidate(source, [3.0, 1.0] + [0.0] * 8 + [-3.0, -1.0] + [0.0] * 8, weight=0.4)


def broad(source: str) -> RuleCandidate:
    """Generic: it speaks everywhere and is a little right each time."""
    return candidate(source, [0.6, 0.4] * 5 + [-0.5] * 10, weight=0.3)


def test_nothing_to_build_from_builds_nothing() -> None:
    builder = create_ruleset_builder()

    assert builder.built([], create_rule_signals(), 1.0, {}, 3, random.Random(1)) == []
    assert builder.built([sharp("a")], (), 1.0, {}, 3, random.Random(1)) == []
    assert builder.built([sharp("a")], create_rule_signals(), 1.0, {}, 0, random.Random(1)) == []


def test_the_first_set_is_an_even_mix_of_every_signal() -> None:
    """A run that builds one set should build the sensible one, not a random corner of the space."""
    built = create_ruleset_builder().built(
        [sharp("a"), broad("b")], create_rule_signals(), 1.0, {}, 1, random.Random(1)
    )

    mix, _ = built[0]
    assert len(set(mix.values())) == 1, "every signal counted the same"
    assert sum(mix.values()) == pytest.approx(1.0)


def test_the_sets_after_it_are_mixed_differently_so_there_is_something_to_race() -> None:
    """**Which aspects synergise is not knowable in advance.** A narrow accurate rule is worth having and
    worth little without generic rules beside it to speak when it does not, so several mixes are built and
    play decides between them."""
    built = create_ruleset_builder().built(
        [sharp("a"), broad("b")], create_rule_signals(), 1.0, {}, 6, random.Random(3)
    )
    mixes = [tuple(sorted(mix.items())) for mix, _ in built]

    assert len(set(mixes)) > 1, "a race needs runners that differ"
    assert all(sum(mix.values()) == pytest.approx(1.0) for mix, _ in built)


def test_a_set_takes_what_it_can_afford_and_no_more() -> None:
    """The packing is the same knapsack the reading does: best value for the cost first, taken while it fits.
    A budget of one rule's cost buys one rule."""
    costs = {"a": RuleCost(0.010, 0.0, 20), "b": RuleCost(0.010, 0.0, 20)}

    built = create_ruleset_builder().built(
        [sharp("a"), broad("b")], create_rule_signals(), 0.012, costs, 1, random.Random(1)
    )

    assert len(built[0][1]) == 1


def test_many_cheap_rules_and_few_dear_ones_are_both_reachable() -> None:
    """Which is the point of packing by ratio rather than by worth: a set of many cheap rules and a set of a
    few expensive ones are different answers to the same budget, and both should be buildable."""
    cheap = [candidate(f"cheap {at}", [float(at % 3)] * 10 + [-1.0] * 10, weight=0.1) for at in range(8)]
    dear = candidate("dear", [3.0, 1.0] * 5 + [-3.0] * 10, weight=0.9)
    costs = {one.rule.source: RuleCost(0.001, 0.0, 20) for one in cheap}
    costs["dear"] = RuleCost(0.020, 0.0, 20)

    built = create_ruleset_builder().built(
        [*cheap, dear], create_rule_signals(), 0.010, costs, 1, random.Random(1)
    )

    taken = [one.rule.source for one in built[0][1]]
    assert len(taken) > 1, "the cheap ones fit where the dear one would not"
    assert "dear" not in taken


def test_a_rule_nobody_has_timed_is_taken_so_that_it_gets_measured() -> None:
    """Priced at nothing until it has been read, which is what stops the packing from starving whatever it
    has never tried."""
    built = create_ruleset_builder().built(
        [sharp("never timed")], create_rule_signals(), 1e-9, {}, 1, random.Random(1)
    )

    assert [one.rule.source for one in built[0][1]] == ["never timed"]
