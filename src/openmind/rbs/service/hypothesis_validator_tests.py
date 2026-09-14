import pytest

from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.model.hypothesis import Hypothesis
from openmind.rbs.service.advantage_contrast import AdvantageContrast
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.consequence_library_tests import strip_domain
from openmind.rbs.service.hypothesis_discoverer_tests import WINS, rows
from openmind.rbs.service.hypothesis_validator import HypothesisValidator
from openmind.rbs.service.primitive_generator_tests import SETTINGS
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


def new_validator() -> HypothesisValidator:
    evaluator = ConditionEvaluator(
        RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build()
    )
    return HypothesisValidator(evaluator, ActionRowMapper(), AdvantageContrast())


def hypothesis(conditions: tuple[PythonRule, ...], direction: int) -> Hypothesis:
    return Hypothesis("place", conditions, (), direction, 0.6 * direction, 10, 80, 1.0, False, 5.0)


def test_a_hypothesis_holding_on_the_validation_states_is_validated_and_the_others_are_not() -> None:
    tests = new_validator().validate(
        strip_domain(),
        (hypothesis((WINS,), 1), hypothesis((WINS,), -1), hypothesis((PythonRule("lamp == 1"),), 1)),
        rows(),
        SETTINGS,
        seed=1,
    )

    assert [(test.validated, test.validation_states) for test in tests] == [(True, 10), (False, 10), (False, 0)]
    assert tests[0].validation_effect == pytest.approx(0.6)
    assert tests[0].p_value <= tests[0].q_value <= 0.05
    assert (tests[2].validation_effect, tests[2].p_value) == (None, 1.0)


def test_the_same_seed_gives_the_same_p_values() -> None:
    hypotheses = (hypothesis((WINS,), 1),)

    first = new_validator().validate(strip_domain(), hypotheses, rows(), SETTINGS, seed=7)
    second = new_validator().validate(strip_domain(), hypotheses, rows(), SETTINGS, seed=7)

    assert first == second
