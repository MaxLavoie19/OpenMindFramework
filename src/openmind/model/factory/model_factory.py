from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.model.service.model_drawer import ModelDrawer
from openmind.model.service.rule_budget import ALLOWANCE, RuleBudget
from openmind.model.model.rule_signal import RuleSignal
from openmind.model.service.rule_admission import RuleAdmission
from openmind.model.service.rule_price import RulePrice
from openmind.model.service.rule_signals import FiresOften, MovedTheFit, SaysSomethingNew, WentWithWinning
from openmind.model.service.model_registry import ModelRegistry
from openmind.model.service.model_timer import ModelTimer


def create_model_registry() -> ModelRegistry:
    """The model registry, with the accuracy scorer measuring the mechanisms its models read through."""
    return ModelRegistry(AccuracyScorer())


def create_model_drawer() -> ModelDrawer:
    """The model drawer, over the registry that holds what fills each task.

    Its own registry rather than one shared with a caller: a registry keeps nothing of its own, taking the
    knowledge base on every call, so two of them are the same thing twice and neither can drift."""
    return ModelDrawer(create_model_registry())


def create_model_timer() -> ModelTimer:
    """The model timer, on wall time."""
    return ModelTimer()



def create_rule_budget() -> RuleBudget:
    """What each signal has earned and spent vouching for rules."""
    return RuleBudget()


def create_rule_price() -> RulePrice:
    """What a rule costs to vouch for, in the bits it takes to write down."""
    return RulePrice()


def create_rule_admission(allowance: float = ALLOWANCE) -> RuleAdmission:
    """What decides which candidate rules are vouched for, on a budget of so many bits a round.

    The allowance is the one knob: a larger one buys more rules and a smaller one fewer. It is a budget in this
    project's sense — the caller's to set, never a number chosen inside a service."""
    return RuleAdmission(RuleBudget(allowance), create_rule_price())


def create_rule_signals() -> tuple[RuleSignal, ...]:
    """The signals that propose rules, all of them.

    **A variety, because both kinds of rule are wanted.** A game needs the specific rule that speaks once and
    decides the game, and the generic one that speaks every position and is a little right each time. Three of
    these reward a term for being sharp where it fired, which is what a specific rule is good at; the fourth
    asks for coverage, which is what nothing else in the economy would ask for."""
    return (WentWithWinning(), MovedTheFit(), SaysSomethingNew(), FiresOften())
