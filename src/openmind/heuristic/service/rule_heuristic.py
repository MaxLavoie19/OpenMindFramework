import json
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.knowledge.constant.rule_kind_constant import MOVE, POSITION
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.constant.rule_constant import RULES_DEFINITIONS
from openmind.rule.service.rule_caller import RuleCaller
from openmind.heuristic.model.node import Node
from openmind.world.model.action import Action
from openmind.world.model.state import State

if TYPE_CHECKING:
    from openmind.rbs.service.consequence_library import ConsequenceLibrary

logger = logging.getLogger(__name__)


class RuleHeuristic:
    """Runs a heuristic ruleset's RBS: what a position is worth to a player, what each move is worth to the player
    taking it, and what each rule added. It keeps nothing: built once, it is given the RBS with every call.

    Each rule's reading is multiplied by its weight in the ruleset that lists it, and the products are summed. A rule
    reading nothing here adds nothing; one that raises, or gives something other than a finite number, is left out, so
    a heuristic survives a rule that doesn't apply."""

    def __init__(
        self,
        rule_caller: RuleCaller,
        consequence_library: "ConsequenceLibrary | None" = None,
        reading: float = 1.0,
    ) -> None:
        self._rule_caller = rule_caller
        self._consequence_library = consequence_library
        #: What share of a ruleset's weight is read before the rest is left alone for this decision. One reads
        #: every rule, which is what this did before.
        #:
        #: **The rules stay in the ruleset; what changes is what is read here and now.** A rule too quiet to
        #: matter in this position may be the rule that decides another one, and taking it out of the ruleset
        #: because it was quiet here is a rule dropped for being specific — the error this project has a
        #: standing rule against. So nothing is removed, and the saving is per decision.
        #:
        #: **Heaviest first, which is the order the fit already writes them in.** Measured on a real fitted
        #: heuristic: three rules of eight carried the whole weight, the other five between 0.00072 and
        #: 0.000036 against 0.494 — fourteen thousand times too small to reorder anything. Reading to 99.9% of
        #: the weight reads three rules instead of eight and cannot shift a value by more than a thousandth of
        #: what the ruleset can say.
        #:
        #: **It is a budget the caller sets and it is approximate, which is worth saying out loud.** A tail
        #: left unread is a tail assumed not to matter, and that assumption is only as good as the share
        #: chosen. One assumes nothing.
        self._reading = max(0.0, min(1.0, reading))

    def value(self, rbs: RuleBasedSystem, node: Node, player: str) -> float | None:
        """What the position is worth to the player; None where the ruleset has no position rule, or none could be
        read. What its rules read besides the state — the consequences of the position — is extracted through the
        node, so every model reading that node shares it."""
        rules = self._weighted(rbs, POSITION)
        if not rules:
            return None
        with process_debugger().frame("evaluation", context=rbs.context, state=node.state, details={"player": player}):
            return self._weighed(rbs, rules, node.state, self._names(node, player))

    def values(self, rbs: RuleBasedSystem, node: Node, players: Sequence[str]) -> tuple[float, ...] | None:
        """Each player's value, in the players' order; None where any of them can't be valued."""
        valued = [self.value(rbs, node, player) for player in players]
        return None if any(value is None for value in valued) else tuple(valued)  # type: ignore[arg-type]

    def rate(
        self, rbs: RuleBasedSystem, node: Node, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]:
        """What each move is worth to the player taking it; None for a move the ruleset says nothing about."""
        rules = self._weighted(rbs, MOVE)
        if not rules:
            return (None,) * len(actions)
        names = self._names(node, player)
        with process_debugger().frame("evaluation", context=rbs.context, state=node.state, details={"moves": len(actions)}):
            return tuple(
                self._weighed(rbs, rules, node.state, names | {"action": action.name} | dict(action.parameters))
                for action in actions
            )

    def explain(self, rbs: RuleBasedSystem, node: Node, player: str) -> tuple[tuple[RuleRecord, float], ...]:
        """Each position rule with what it adds to the player's value: its weight times its reading."""
        return self._readings(rbs, self._weighted(rbs, POSITION), node.state, self._names(node, player))

    def describe(self, rbs: RuleBasedSystem) -> str:
        """The RBS as the heuristics it judges with: its context and every position and move rule with its weight
        and what the rule reads. Two RBSs describe alike when they judge alike.

        **What a rule reads is part of the description, because a name need not be one.** A fitted rule is named
        for its own expression and reads as itself; a rule somebody wrote is named `a mark on the middle cell`,
        which says what it is for and not what it does. A description is what builds the model again and what a
        person reads to see what it believed, and a label does neither. It goes last so anything reading the
        name and the weight is unaffected."""
        return json.dumps(
            {
                "context": rbs.context,
                "position": [self._said(rule, weight) for rule, weight in self._weighted(rbs, POSITION)],
                "move": [self._said(rule, weight) for rule, weight in self._weighted(rbs, MOVE)],
            },
            indent=2,
        )

    def _said(self, rule: RuleRecord, weight: float) -> list[object]:
        """One rule as a description lists it: what it is called, what it weighs, and what it reads."""
        return [rule.name, weight, str(getattr(rule.rule, "source", "") or "")]

    def _weighted(self, rbs: RuleBasedSystem, kind: str) -> tuple[tuple[RuleRecord, float], ...]:
        """That kind's rules, heaviest first, down to the share of the weight this reads.

        A rule left out here is left out of *this reading* and stays in the ruleset, so the next position asks
        the whole set again. Ties and a share of one both read everything."""
        found = tuple((rule, weight) for rule, weight in rbs.rules if rule.kind == kind)
        if self._reading >= 1.0 or len(found) < 2:
            return found
        ordered = sorted(found, key=lambda one: -abs(one[1]))
        whole = math.fsum(abs(weight) for _, weight in ordered)
        if whole <= 0.0:
            return found
        wanted, running = self._reading * whole, 0.0
        for at, (_, weight) in enumerate(ordered):
            running += abs(weight)
            if running >= wanted:
                return tuple(ordered[: at + 1])
        return tuple(ordered)

    def _names(self, node: Node, player: str) -> dict[str, object]:
        """What a heuristic reads besides the state's models: `me`, `other`, `win_chance`, `wins`, `near`, `here`.
        Extracted through the node the first time a model asks for them, and shared from then on."""
        library = self._consequence_library
        if library is None or node.game is None:
            return {"me": player}
        extracted = node.feature(f"consequences of {player}", lambda: library.names(node.game, node.state, player))  # type: ignore[arg-type]
        return extracted  # type: ignore[return-value]

    def _weighed(
        self, rbs: RuleBasedSystem, rules: Sequence[tuple[RuleRecord, float]], state: State, names: dict[str, object]
    ) -> float | None:
        readings = self._readings(rbs, rules, state, names)
        return math.fsum(added for _, added in readings) if readings else None

    def _readings(
        self, rbs: RuleBasedSystem, rules: Sequence[tuple[RuleRecord, float]], state: State, names: dict[str, object]
    ) -> tuple[tuple[RuleRecord, float], ...]:
        added: list[tuple[RuleRecord, float]] = []
        definitions = rbs.definitions(RULES_DEFINITIONS)
        for rule, weight in rules:
            try:
                read = self._rule_caller.value(rule.rule, state, None, names, definitions)
            except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                continue
            if read is None:
                added.append((rule, 0.0))
                continue
            if not isinstance(read, bool | int | float | np.bool_ | np.number):
                continue
            reading = float(read)  # type: ignore[arg-type]
            if math.isfinite(reading):
                added.append((rule, weight * reading))
        return tuple(added)
