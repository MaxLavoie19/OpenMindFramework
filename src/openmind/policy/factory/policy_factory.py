import random

from openmind.policy.service.random_picker import RandomPicker
from openmind.policy.service.rated_policy_picker import RatedPolicyPicker
from openmind.policy.service.rule_optimizer import RuleOptimizer
from openmind.policy.service.rule_policy_valuer import RulePolicyValuer
from openmind.policy.service.solver_optimizer import SolverOptimizer
from openmind.rule.factory.rule_factory import create_rule_caller


def create_random_picker(source: random.Random | None = None) -> RandomPicker:
    """The optimizer picking a legal action at random, its random source given so a run can be repeated."""
    return RandomPicker(source)


def create_solver_optimizer() -> SolverOptimizer:
    """The optimizer letting the constraint solver work the values out."""
    return SolverOptimizer()


def create_rule_optimizer() -> RuleOptimizer:
    """The optimizer running a ruleset that computes the action."""
    return RuleOptimizer(create_rule_caller())


def create_rule_policy_valuer() -> RulePolicyValuer:
    """The policy value model running a ruleset."""
    return RulePolicyValuer(create_rule_caller())


def create_rated_policy_picker(policy_valuer: object | None = None) -> RatedPolicyPicker:
    """The policy picker reading what a policy value model says; the rule-based one unless another is given."""
    return RatedPolicyPicker(create_rule_policy_valuer() if policy_valuer is None else policy_valuer)  # type: ignore[arg-type]
