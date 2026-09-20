from collections.abc import Callable

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.policy.factory.policy_factory import create_rated_policy_picker, create_rule_policy_valuer
from openmind.policy.service.rated_policy_picker import RatedPolicyPicker
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]

START = State.of(danger=2, turn="me")


class Rating:
    """A policy value model saying what a test set out, by policy name."""

    def __init__(self, **ratings: float | None) -> None:
        self._ratings = ratings

    def rate(self, model: object, node: Node, policies: tuple[Policy, ...], player: str) -> tuple[float | None, ...]:
        return tuple(self._ratings.get(policy.name or "default") for policy in policies)


def policies(knowledge: KnowledgeBase, *names: str) -> tuple[Policy, ...]:
    context = knowledge.ensure_context("soldier").id
    return tuple(knowledge.policy(Policy(name, context, f"do {name}" if name else "")) for name in names)


def test_the_policies_worth_expanding_come_back_best_first(knowledge: KnowledgeBase) -> None:
    picked = RatedPolicyPicker(Rating(flee=0.2, charge=0.9, default=0.1))

    worth = picked.pick(None, Node(START), policies(knowledge, "", "flee", "charge"), "me")

    assert [policy.name for policy in worth] == ["charge", "flee"]


def test_a_policy_worth_less_than_picking_at_random_is_not_worth_expanding(knowledge: KnowledgeBase) -> None:
    picked = RatedPolicyPicker(Rating(flee=0.05, charge=0.9, default=0.5))

    worth = picked.pick(None, Node(START), policies(knowledge, "", "flee", "charge"), "me")

    assert [policy.name for policy in worth] == ["charge"]


def test_where_picking_at_random_is_the_best_rated_nothing_else_is_worth_considering(knowledge: KnowledgeBase) -> None:
    picked = RatedPolicyPicker(Rating(flee=0.1, charge=0.2, default=0.9))

    worth = picked.pick(None, Node(START), policies(knowledge, "", "flee", "charge"), "me")

    assert [policy.name for policy in worth] == [""]


def test_a_policy_nothing_is_known_about_is_kept_last_rather_than_dismissed(knowledge: KnowledgeBase) -> None:
    picked = RatedPolicyPicker(Rating(charge=0.9, default=0.1))

    worth = picked.pick(None, Node(START), policies(knowledge, "", "flee", "charge"), "me")

    assert [policy.name for policy in worth] == ["charge", "flee"]


def test_without_a_policy_nothing_is_picked(knowledge: KnowledgeBase) -> None:
    assert create_rated_policy_picker().pick(None, Node(START), (), "me") == ()


def test_a_rule_based_policy_value_model_rates_by_its_rules(game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]) -> None:
    from openmind.knowledge.constant.task_constant import POLICY_VALUE
    from openmind.rbs.factory.rbs_factory import create_rule_based_system
    from openmind.rule.model.python_rule import PythonRule

    game("tictactoe")
    heuristic(
        "tictactoe",
        "fleeing is worth it when in danger",
        PythonRule("danger if sub_goal == 'do flee' else 0"),
        0.5,
        task=POLICY_VALUE,
    )
    rules = create_rule_based_system(knowledge, "tictactoe", POLICY_VALUE)
    flee, charge = policies(knowledge, "flee", "charge")

    rated = create_rule_policy_valuer().rate(rules, Node(START), (flee, charge), "me")

    assert rated == (1.0, 0.0)
