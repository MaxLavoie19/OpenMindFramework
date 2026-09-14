from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.mapper.hypothesis_text_mapper import HypothesisTextMapper
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.service.advantage_contrast import AdvantageContrast
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.coverage_filter import CoverageFilter
from openmind.rbs.service.goal_pattern_miner import GoalPatternMiner
from openmind.rbs.service.hypothesis_discoverer import HypothesisDiscoverer
from openmind.rbs.service.hypothesis_validator import HypothesisValidator
from openmind.rbs.service.primitive_generator import PrimitiveGenerator
from openmind.rbs.service.rule_generator import RuleGenerator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class RuleGeneratorBuilder:
    """Wires a rule generator: one consequence library shared by its miner, primitive generator and condition
    evaluator."""

    def build(self) -> RuleGenerator:
        names = VariableNameMapper()
        state_namespace_mapper = StateNamespaceMapper(names)
        library = ConsequenceLibraryBuilder().build()
        evaluator = ConditionEvaluator(RuleCompiler(), RuleRunner(state_namespace_mapper), library)
        row_mapper, contrast = ActionRowMapper(), AdvantageContrast()
        return RuleGenerator(
            row_mapper,
            GoalPatternMiner(library, SolverBuilder().build(), state_namespace_mapper, names),
            PrimitiveGenerator(library, evaluator, state_namespace_mapper, names),
            HypothesisDiscoverer(evaluator, row_mapper, contrast),
            HypothesisValidator(evaluator, row_mapper, contrast),
            CoverageFilter(evaluator),
            RuleTextMapper(),
            HypothesisTextMapper(),
        )
