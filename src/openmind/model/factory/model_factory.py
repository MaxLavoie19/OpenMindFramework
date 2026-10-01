from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.model.service.model_drawer import ModelDrawer
from openmind.model.service.rule_budget import ALLOWANCE, RuleBudget
from openmind.model.model.rule_signal import RuleSignal
from openmind.model.service.rule_admission import RuleAdmission
from openmind.model.service.ruleset_builder import RulesetBuilder
from openmind.model.service.model_retirement import ModelRetirement
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
    asks for coverage, which is what nothing else in the economy would ask for.

    **None of them asks a teller, and that is a decision rather than an omission.** A teller grades a
    position, so only something that produces a position value can be compared with what it says — and a
    single term does not produce one. There was a fifth signal here that asked how much a heuristic's
    agreement with a teller owed to each of its terms, and what it kept running into were the symptoms of
    that mismatch: a term with no variance over the rows it fires on has nothing to correlate, and the rows a
    detector is silent on are most of them. The teller's question belongs one level up, where a whole
    heuristic reads the positions and `TellerJudging` asks how closely that ordering follows the teller's.
    Outside knowledge is a claim with measured reliability either way; what changed is the unit it is a claim
    about."""
    return WentWithWinning(), MovedTheFit(), SaysSomethingNew(), FiresOften()


def create_model_retirement() -> ModelRetirement:
    """What decides that a heuristic is no longer worth asking.

    Retired and not deleted: the store is append-only and a model is a record with beliefs about it, so
    retiring is believing it is not worth asking. What was learned about it survives, and later evidence can
    un-retire it."""
    return ModelRetirement()


def create_ruleset_builder() -> RulesetBuilder:
    """What builds candidate rulesets out of what the signals make of the rules.

    Signals are aspects rather than authors: a set is a mix of them, several mixes are built, and which mix
    synergises is settled by playing them against one another rather than decided here."""
    return RulesetBuilder()
