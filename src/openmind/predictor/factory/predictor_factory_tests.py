from openmind.predictor.factory.predictor_factory import create_effects_runner, create_rule_predictor
from openmind.predictor.service.drawn_effects import DrawnEffects
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.statement.model.consequence import Consequence
from openmind.rule.model.consequence_rule import ConsequenceRule
from openmind.statement.model.drawn import other
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.state import State

PLAYERS = ("first", "second")


def a_board() -> State:
    return State.of(grid=Grid.of([[None, None], [None, None]]), turn="first")


def passing() -> ConsequenceRule:
    """What was learned of an action: the turn becomes the other player's."""
    return ConsequenceRule((Consequence("Told", "place", "turn", value=other()),))


def test_a_runner_given_a_caller_uses_it_rather_than_making_its_own():
    """A factory that builds its dependency cannot be told anything, and what a learned action does is exactly
    where a game needs its own: a consequence may say *the player not acting*, and which of a position's models
    says who is acting is the game's to tell. Built its own, a learned game played every move through a caller
    that could not read its turn, and its turn never passed."""
    caller = create_rule_caller(
        consequence_caller=DrawnEffects(acting=lambda state: state.value("turn"), players=PLAYERS)
    )

    after = create_effects_runner(caller).run(
        a_board(), Action("place", ()), ((1.0, passing()),)
    ).outcomes[0][0]

    assert after.value("turn") == "second"


def test_a_runner_given_nothing_still_runs_the_rules_a_project_wrote():
    """The default has to go on working: most games are declared, and nothing about them needs a learned caller."""
    from openmind.rule.model.python_rule import PythonRule

    after = create_effects_runner().run(a_board(), Action("place", ()), ((1.0, PythonRule("turn = 'second'")),))

    assert after.outcomes[0][0].value("turn") == "second"


def test_a_predictor_hands_its_caller_on_to_the_runner_beneath_it():
    """The caller has to reach the bottom. A predictor that took one and kept it to itself would look wired and
    behave exactly as it did before, which is the failure worth pinning — so this asks it to predict rather
    than asking it what it holds."""
    from openmind.knowledge.constant.rule_kind_constant import EFFECTS
    from openmind.knowledge.constant.task_constant import SIMULATION
    from openmind.knowledge.model.rule_record import RuleRecord
    from openmind.knowledge.model.ruleset import Ruleset
    from openmind.knowledge.model.source import Source
    from openmind.rbs.model.rule_based_system import RuleBasedSystem

    source = Source("a test")
    rbs = RuleBasedSystem(
        "a game",
        "context-1",
        Ruleset(SIMULATION, "context-1", SIMULATION, source),
        ((RuleRecord("what place leads to", EFFECTS, passing(), source, action="place", id="one"), 1.0),),
    )
    caller = create_rule_caller(
        consequence_caller=DrawnEffects(acting=lambda state: state.value("turn"), players=PLAYERS)
    )

    predicted = create_rule_predictor(caller).predict_action(rbs, a_board(), Action("place", ()))

    assert predicted.outcomes[0][0].value("turn") == "second"
