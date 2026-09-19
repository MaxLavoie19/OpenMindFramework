from openmind.csp.factory.csp_factory import create_solver
from openmind.game.service.game_registry import GameRegistry
from openmind.knowledge.constant.knowledge_constant import MOVE_VALUE, POSITION_VALUE, SIMULATION
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.predictor.factory.predictor_factory import create_rule_predictor
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.simulation import Simulation
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.rule.factory.rule_factory import create_rule_caller


def create_value_generator(workers: int = 1) -> ValueGenerator:
    """A value generator with its term generator, term evaluator and sparse fitter, evaluating terms in that many worker
    processes."""
    return ValueGeneratorBuilder().with_workers(workers).build()


def create_simulation() -> Simulation:
    """The simulation service, with its solver and its solution cache, the rule predictor and a rule caller. Build it
    once and give it to whatever runs a simulation."""
    return Simulation(create_solver(), create_rule_predictor(), create_rule_caller())


def create_rule_based_system(knowledge_base: KnowledgeBase, context: str, task: str = SIMULATION) -> RuleBasedSystem:
    """The RBS of the context's ruleset for that task, named as applications name the context: that ruleset's rules
    with their weights there. A context without one takes the ruleset of a context it inherits from, nearest first; with
    none at all, raises ValueError."""
    rbs = find_rule_based_system(knowledge_base, context, task)
    if rbs is None:
        raise ValueError(f"{context} has no {task} ruleset")
    return rbs


def find_rule_based_system(knowledge_base: KnowledgeBase, context: str, task: str) -> RuleBasedSystem | None:
    """The RBS `create_rule_based_system` gives, or None where there is none."""
    known = knowledge_base.context_named(context)
    if known is None:
        return None
    ruleset = context_ruleset(knowledge_base, known.id, task)
    if ruleset is None:
        return None
    return RuleBasedSystem(context, known.id, ruleset, knowledge_base.ruleset_rules(ruleset.id))


def create_rule_based_game(
    knowledge_base: KnowledgeBase, context: str, simulation: Simulation | None = None
) -> RuleBasedGame:
    """The temporary facade over a context's RBSs: its simulation's and its position and move value heuristics', each
    taken from a context it inherits from where it has none. A context the knowledge base holds no ruleset for gives a
    facade that can't say where the game starts."""
    known = knowledge_base.context_named(context)
    heuristics = tuple(
        rbs for task in (POSITION_VALUE, MOVE_VALUE) if (rbs := find_rule_based_system(knowledge_base, context, task))
    )
    return RuleBasedGame(
        context,
        find_rule_based_system(knowledge_base, context, SIMULATION),
        heuristics,
        create_simulation() if simulation is None else simulation,
        create_rule_caller(),
        ConsequenceLibraryBuilder().build(),
        None if known is None else known.id,
    )


def create_game(name: str, knowledge_base: KnowledgeBase, registry: GameRegistry | None = None) -> RuleBasedGame:
    """The facade for a registered game, or "game/variant", its rules declared into the knowledge base first."""
    return create_rule_based_game(knowledge_base, (registry or GameRegistry()).declare(name, knowledge_base))


def context_ruleset(knowledge_base: KnowledgeBase, context_id: str, task: str) -> Ruleset | None:
    """The context's first ruleset for the task, or else that of the contexts it inherits from, nearest first."""
    seen: set[str] = set()
    pending = [context_id]
    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        own = knowledge_base.rulesets(current, task)
        if own:
            return own[0]
        context = knowledge_base.context_by_id(current)
        if context is not None:
            pending.extend(context.inherits)
    return None
