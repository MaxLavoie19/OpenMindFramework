from openmind.game.service.game_declarer import GameDeclarer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.predictor.factory.predictor_factory import create_rule_predictor
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players
from openmind.world.model.state import State

START = State.of(hand=Map.of({"A": None, "B": None}), payoff=Map.of({"A": None, "B": None}))


def throwing(knowledge: KnowledgeBase) -> RuleBasedSystem:
    """Two players throw a number at once; then the higher throw wins."""
    declarer = GameDeclarer(knowledge, "throw")
    declarer.starts_at(START)
    declarer.played_by(Players(("A", "B"), "payoff"))
    declarer.definitions(PythonRule("WIN, LOSS = 1.0, 0.0"), effects=True)
    declarer.leads_to("throw", PythonRule("hand = hand.with_item(player, number)"))
    declarer.together(
        PythonRule(
            "high = 'A' if hand['A'] > hand['B'] else 'B'\n"
            "payoff = payoff.with_item(high, WIN).with_item('B' if high == 'A' else 'A', LOSS)"
        )
    )
    return create_rule_based_system(knowledge, declarer.done())


def test_the_ruleset_s_effects_run_for_every_action_taken_at_once_then_what_they_lead_to_together(
    knowledge: KnowledgeBase,
) -> None:
    joint = JointAction((("A", Action("throw", (("number", 3),))), ("B", Action("throw", (("number", 5),)))))

    ((outcome, probability),) = create_rule_predictor().predict(throwing(knowledge), START, joint).outcomes

    assert probability == 1.0
    assert outcome.model("payoff") == Map.of({"A": 0.0, "B": 1.0})


def test_one_action_runs_without_a_player_where_its_effects_read_none(knowledge: KnowledgeBase) -> None:
    declarer = GameDeclarer(knowledge, "count")
    declarer.leads_to("add", PythonRule("total = total + step"))
    rbs = create_rule_based_system(knowledge, declarer.done())

    outcomes = create_rule_predictor().predict_action(rbs, State.of(total=1), Action("add", (("step", 2),)))

    assert outcomes.outcomes == ((State.of(total=3), 1.0),)
