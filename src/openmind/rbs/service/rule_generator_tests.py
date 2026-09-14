import logging
from dataclasses import replace

import pytest

from openmind.rbs.factory.rbs_factory import create_rule_generator
from openmind.rbs.service.consequence_library_tests import strip_domain
from openmind.rbs.service.hypothesis_discoverer_tests import WINS, samples
from openmind.rbs.service.primitive_generator_tests import SETTINGS


def test_generation_keeps_the_hypotheses_that_hold_on_the_validation_games(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.rbs")
    # The strip has 12 validation states for dozens of hypotheses: a 1,000-permutation test can't get a single p-value
    # low enough for Benjamini-Hochberg at 0.05, so this wiring test uses more permutations and a looser rate.
    settings = replace(SETTINGS, permutations=5000, false_discovery_rate=0.2)

    result = create_rule_generator().generate(strip_domain(), samples(), samples(), settings, seed=1)

    rules = result.rule_base.rules
    assert (result.rule_base.domain, rules[0].action, rules[0].conditions) == ("strip", "place", ())
    assert any(rule.conditions == (WINS,) and rule.expected_value == pytest.approx(1.0) for rule in rules)
    kept = {rule.conditions for rule in rules[1:]}
    covered = {coverage.rule.conditions for coverage in result.covered}
    assert kept.isdisjoint(covered)
    assert kept | covered == {test.conditions for test in result.hypotheses if test.validated}
    assert all(coverage.covering in rules for coverage in result.covered)
    assert any(message.startswith("place: kept ") for message in caplog.messages)
    assert result.patterns
    assert any(message.startswith("Validated ") for message in caplog.messages)


def test_without_coverage_every_validated_hypothesis_becomes_a_rule() -> None:
    settings = replace(SETTINGS, permutations=5000, false_discovery_rate=0.2, coverage=False)

    result = create_rule_generator().generate(strip_domain(), samples(), samples(), settings, seed=1)

    assert result.covered == ()
    assert [rule.conditions for rule in result.rule_base.rules[1:]] == [
        test.conditions for test in result.hypotheses if test.validated
    ]
