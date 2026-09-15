import logging
import math

import pytest

from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.mechanics_tests import new_mechanics
from openmind.inference.service.position_view_tests import capture_domain, capture_position
from openmind.rbs.factory.rbs_factory import create_rule_valuer
from openmind.training.service.heuristic_deducer import HeuristicDeducer
from openmind.training.service.signal_preparer import SignalPreparer
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")


def new_preparer() -> SignalPreparer:
    return SignalPreparer(HeuristicDeducer(ExpressionGenerator(VariableNameMapper()), new_mechanics()))


def test_the_library_records_every_deduced_signal_and_holds_the_deduced_bases(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    domain = capture_domain()

    library = new_preparer().prepare(domain)

    names = [record.signal.name for record in library.records]
    assert (len(names), names[0], names[-1], library.supports) == (12, "options", "hanging cell", ())
    assert all(record.agreements + record.disagreements + record.games == 0 for record in library.records)
    assert [name for name, _ in library.value_bases] == ["deduced", *(f"deduced, {name} doubled" for name in names)]
    deduced, options_doubled = library.value_bases[0][1], library.value_bases[1][1]
    assert (deduced.bias, deduced.low, deduced.high) == (0.0, 0.0, 1.0)
    assert [rule.weight for rule in deduced.rules] == pytest.approx([1 / 12] * 12)
    assert [rule.weight for rule in options_doubled.rules] == pytest.approx([2 / 12] + [1 / 12] * 11)
    assert "Prepared 12 signals deduced from the rules: value base deduced, every weight 0.08333333333333333, and 12 variations doubling one weight each" in caplog.messages


def test_the_deduced_base_values_positions_blank_signals_adding_nothing() -> None:
    domain = capture_domain()
    (_, deduced), *_ = new_preparer().prepare(domain).value_bases

    values = create_rule_valuer(deduced, domain).value(capture_position({2: "A", 6: "A", 4: "B", 7: "B"}, "A"))
    start = create_rule_valuer(deduced, domain).value(domain.initial_state)

    assert values is not None and start is not None and all(0.0 < value < 1.0 for value in (*values, *start))
    # At the start only the options signals read anything, and they cancel out between the players' own readings.
    assert start[0] == pytest.approx(1.0 / (1.0 + math.exp(-(0.5 - 0.5) / 12)))
