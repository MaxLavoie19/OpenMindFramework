import logging
from collections.abc import Mapping

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.heuristic.model.node import Node
from openmind.knowledge.constant.rule_kind_constant import ABSTRACTION
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.constant.rule_constant import RULES_DEFINITIONS
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class RuleAbstractor:
    """Runs an abstraction ruleset's RBS: the state as one level of the hierarchy sees it.

    Each abstraction rule gives the models that level holds, by name — `{"economy": state.model("resources")}` — and
    what the level sees is what its rules gave together, a later rule replacing a model an earlier one gave. A rule
    reading nothing, or raising, or giving anything but names and models, gives nothing, so a level survives a rule
    that doesn't apply here. A ruleset whose rules all gave nothing abstracts nothing, and the level sees the state as
    it is.

    It keeps nothing: built once, it is given the RBS with every call."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def abstract(self, rbs: RuleBasedSystem, node: Node, context: str) -> State | None:
        """The state the level sees; None where the ruleset has no abstraction rule, or none could be read."""
        rules = rbs.of(ABSTRACTION)
        if not rules:
            return None
        definitions = rbs.definitions(RULES_DEFINITIONS)
        seen: dict[str, object] = {}
        with process_debugger().frame("abstraction", context=rbs.context, state=node.state, details={"level": context}):
            for rule in rules:
                try:
                    read = self._rule_caller.value(rule.rule, node.state, None, {"context": context}, definitions)
                except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                    continue
                if isinstance(read, Mapping):
                    seen.update({str(name): model for name, model in read.items()})
        if not seen:
            return None
        logger.debug("%s sees %s", context, ", ".join(sorted(seen)))
        return State.of(**seen)  # type: ignore[arg-type]
