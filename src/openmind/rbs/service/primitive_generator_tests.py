from openmind.parallel.service.task_runner import TaskRunner
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.consequence_library_tests import place, position, strip_domain
from openmind.rbs.service.primitive_generator import PrimitiveGenerator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

SETTINGS = GenerationSettings(
    min_visits=1,
    max_conditions=2,
    min_rule_visits=1,
    min_gain=0.05,
    confidence=0.95,
    beam_width=20,
    max_offset=2,
    solo_limit=2,
    patterns=10,
    false_discovery_rate=0.05,
    permutations=1000,
)


def new_generator() -> PrimitiveGenerator:
    names = VariableNameMapper()
    state_namespace_mapper = StateNamespaceMapper(names)
    library = ConsequenceLibraryBuilder().build()
    evaluator = ConditionEvaluator(RuleCompiler(), RuleRunner(state_namespace_mapper), library, TaskRunner(1))
    return PrimitiveGenerator(library, evaluator, state_namespace_mapper, names)


def test_primitives_cover_variables_parameters_offsets_patterns_and_consequences() -> None:
    rows = (
        ActionRow(position({1: "X"}, "X"), place(2), 10, 1.0, 0.0),
        ActionRow(position({1: "X"}, "X"), place(3), 10, 0.2, -0.8),
        ActionRow(position({1: "X", 3: "O"}, "X"), place(2), 10, 1.0, 0.0),
        ActionRow(position({1: "X", 3: "O"}, "X"), place(4), 10, 0.1, -0.9),
    )
    pattern = GoalPattern("place", (PythonRule("near(action, 0, -3) == me"),), 2)

    sources = {rule.source for rule in new_generator().primitives(strip_domain(), rows, (pattern,), SETTINGS)}

    assert {
        "cell[1, 1] == me",
        "cell[1, 1] == other",
        "cell[1, 3] == None",
        "turn == 'X'",
        "col == 2",
        "near(action, 0, -1) == me",
        "near(action, 0, -1) == None",
        "near(action, 0, 2) == OUTSIDE",
        "near(action, 0, -3) == me",
        "win_chance(action) >= 1",
        "win_chance(action) <= 0",
        "wins(me, action) - wins(me) <= -1",
    } <= sources
