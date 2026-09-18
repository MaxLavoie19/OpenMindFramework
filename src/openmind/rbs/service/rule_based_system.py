import json
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.csp.service.solver import Solver
from openmind.doxastic.constant.rule_kind_constant import (
    CONSTRAINT,
    DEFINITIONS,
    EFFECTS,
    EMPTY,
    ENDING,
    INITIAL,
    MOVE,
    PICTURE,
    PLAYERS,
    POSITION,
    RECORD,
    TIMEOUT,
    VALUES,
)
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.predictor.service.predictor import Effects, Predictor
from openmind.rbs.constant.game_record_constant import ACTIONS, FLAGGED_PLAYER, PAYOFFS
from openmind.rbs.constant.rule_based_constant import EFFECTS_DEFINITIONS, RULES_DEFINITIONS
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value
from openmind.world.service.state_reader import StateReader

if TYPE_CHECKING:
    from openmind.rbs.service.consequence_library import ConsequenceLibrary

logger = logging.getLogger(__name__)

#: The state a rule that reads nothing is called with.
NOTHING = State(())


class RuleBasedSystem:
    """One game, as the rules the knowledge base holds for a context.

    A chess RBS holds chess's rules: what makes a move legal, what a move leads to, where the game starts, who plays,
    when it is over, and the heuristics that judge a position and a move. It answers with them, and the CSP is its
    solver for legal moves. A relaxation is another context holding the same rules less some constraints, so a relaxed
    game is an RBS like any other.

    Nothing about a game is stored as a game: the rules are what is kept, and an RBS is what retrieval makes of them.
    `actions`, `outcomes`, `value` and `rate` are roles a learned model can fill instead; this is the one that fills
    them with rules."""

    def __init__(
        self,
        context: str,
        rules: tuple[RuleRecord, ...],
        solver: Solver,
        predictor: Predictor,
        rule_caller: RuleCaller,
        state_reader: StateReader,
        consequence_library: "ConsequenceLibrary | None" = None,
    ) -> None:
        self._context = context
        self._rules = rules
        self._solver = solver
        self._predictor = predictor
        self._rule_caller = rule_caller
        self._state_reader = state_reader
        self._consequence_library = consequence_library
        self._by_kind: dict[str, list[RuleRecord]] = {}
        for rule in rules:
            self._by_kind.setdefault(rule.kind, []).append(rule)
        self._action_names = tuple(dict.fromkeys(rule.action for rule in self._of(CONSTRAINT, VALUES) if rule.action))
        self._start: State | None = None
        self._players: Players | None = None

    @property
    def context(self) -> str:
        """The game the rules were retrieved for; a relaxation's context is its own."""
        return self._context

    @property
    def rules(self) -> tuple[RuleRecord, ...]:
        """Every rule retrieved, heaviest first."""
        return self._rules

    def start(self) -> State:
        """Where the game starts; read once, since the rules don't change under an RBS."""
        if self._start is None:
            read = self._read(INITIAL, "where the game starts")
            self._start = read if isinstance(read, State) else State(tuple(read))  # type: ignore[arg-type]
        return self._start

    def players(self) -> Players:
        """Who plays, the variable naming the player to act, and each player's payoff variable; read once."""
        if self._players is None:
            read = self._read(PLAYERS, "who plays")
            if isinstance(read, Players):
                self._players = read
            else:
                names, to_act, payoffs = read  # type: ignore[misc]
                self._players = Players(tuple(names), to_act, tuple(payoffs))
        return self._players

    def empty(self, base: str) -> Value:
        """What a cell of that grid holds when nothing is on it; None where the game doesn't say."""
        for rule in self._by_kind.get(EMPTY, ()):
            if rule.parameter == base:
                return self._rule_caller.value(rule.rule, NOTHING)  # type: ignore[return-value]
        return None

    def empties(self) -> tuple[tuple[str, Value], ...]:
        """Each grid's empty value by base, sorted by base."""
        found = {rule.parameter: self._rule_caller.value(rule.rule, NOTHING) for rule in self._by_kind.get(EMPTY, ()) if rule.parameter}
        return tuple(sorted(found.items()))  # type: ignore[arg-type]

    def actions(self, state: State, limit: int | None = None, player: str | None = None) -> tuple[Action, ...]:
        """Every legal action in the state, solved by the CSP over the constraint rules, at most limit of them. Given a
        player, such as one of several players acting at once, the rules also read it as `player`."""
        return self.actions_with_statistics(state, limit, player)[0]

    def actions_with_statistics(
        self, state: State, limit: int | None = None, player: str | None = None
    ) -> tuple[tuple[Action, ...], SolveStatistics]:
        """The legal actions `actions` gives, with what the search did, summed over the game's actions."""
        found: list[Action] = []
        assignments = dead_ends = pruned_values = 0
        for name in self._action_names:
            remaining = None if limit is None else limit - len(found)
            if remaining == 0:
                break
            solved, statistics = self._solver.solve_with_statistics(
                state, name, self._values(name), self._constraints(name), self._definitions(RULES_DEFINITIONS), remaining, player
            )
            found.extend(solved)
            assignments += statistics.assignments
            dead_ends += statistics.dead_ends
            pruned_values += statistics.pruned_values
        return tuple(found), SolveStatistics(len(found), assignments, dead_ends, pruned_values)

    def joint_actions(self, state: State) -> tuple[tuple[int, tuple[Action, ...]], ...]:
        """Each player to act, by index in the players' names, with its legal actions, the rules reading that player as
        `player`: what players acting at once choose from. Empty when the game is over, no player to act having any.
        Players to act without a legal action while others have one raise ValueError."""
        players = self.players()
        legal = tuple(
            (index, self.actions(state, player=players.names[index]))
            for index in self._state_reader.players_to_act(state, players)
        )
        stuck = [players.names[index] for index, actions in legal if not actions]
        if not stuck:
            return legal
        if len(stuck) == len(legal):
            return ()
        raise ValueError(
            f"{', '.join(stuck)} can't act while other players to act can: every player to act needs an action"
        )

    def outcomes(self, state: State, action: Action) -> OutcomeDistribution:
        """What the action leads to, with each outcome's chance."""
        return self._predictor.predict(state, action, self._effects(action.name), self._definitions(EFFECTS_DEFINITIONS))

    def joint_outcomes(self, state: State, joint: JointAction) -> OutcomeDistribution:
        """What the players' actions, taken at once, lead to together."""
        effects = {name: self._effects(name) for name in self._action_names}
        together = tuple((rule.probability, rule.rule) for rule in self._by_kind.get(EFFECTS, ()) if rule.action is None)
        return self._predictor.predict_joint(state, joint, effects, together, self._definitions(EFFECTS_DEFINITIONS))

    def ended(self, state: State) -> str | None:
        """Why the game ended, or None while it goes on and where the game doesn't say."""
        value = self._call(ENDING, state)
        return None if value is None else str(value)

    def record(
        self, actions: Sequence[Action], flagged: str | None = None, payoffs: Sequence[float] | None = None
    ) -> str | None:
        """The game's record, read from where the game starts and the actions played. The rule is also given the player
        whose time ran out, `flagged` (None when none did), and the final `payoffs` (None when unknown), since a game
        ended on time can't be told from its moves. None where the game doesn't record itself."""
        if not self._by_kind.get(RECORD):
            return None
        parameters = {
            ACTIONS: tuple(actions),
            FLAGGED_PLAYER: flagged,
            PAYOFFS: None if payoffs is None else tuple(payoffs),
        }
        value = self._call(RECORD, self.start(), parameters)
        return None if value is None else str(value)

    def flagged(self, state: State, **parameters: object) -> State | None:
        """The state after a player's clock ran out, the player read as `flagged`; None where the game can't be played
        on a clock."""
        rules = self._by_kind.get(TIMEOUT, ())
        if not rules:
            return None
        return self._rule_caller.apply(rules[0].rule, state, parameters, self._definitions(EFFECTS_DEFINITIONS))  # type: ignore[arg-type]

    def timed(self) -> bool:
        """Whether the game says what running out of time does, without which it can't be played on a clock."""
        return bool(self._by_kind.get(TIMEOUT))

    def picture(self, state: State, **parameters: object) -> str | None:
        """The position as an SVG image, or None where the game doesn't draw itself."""
        return self._call(PICTURE, state, parameters)  # type: ignore[return-value]

    def value(self, state: State, player: str) -> float | None:
        """What the position is worth to the player, by the position heuristics retrieved for this context: each rule's
        reading times its weight here, summed. None where the context has no such rule, or where none could be read."""
        rules = self._by_kind.get(POSITION, ())
        return self._weighed(rules, state, self._names(state, player)) if rules else None

    def values(self, state: State) -> tuple[float, ...] | None:
        """Each player's value, in the order of the players' names; None where any of them can't be valued."""
        valued = [self.value(state, player) for player in self.players().names]
        return None if any(value is None for value in valued) else tuple(valued)  # type: ignore[arg-type]

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        """What each move is worth to the player to act, by the move heuristics retrieved for this context: each
        rule's reading times its weight here, summed. None for a move where the context has no such rule, or where
        none could be read. This is the RBS filling the role of a rater."""
        rules = self._by_kind.get(MOVE, ())
        if not rules:
            return (None,) * len(actions)
        player = self.players().names[self._state_reader.player_to_act(state, self.players())]
        names = self._names(state, player)
        return tuple(
            self._weighed(rules, state, names | {"action": action.name} | dict(action.parameters)) for action in actions
        )

    def describe(self) -> str:
        """The RBS as the heuristics it judges with: its context and every position and move rule with its weight
        there. Two RBSs describe alike when they judge alike, whatever else their context holds."""
        return json.dumps(
            {
                "context": self._context,
                "position": [[rule.name, rule.weight(self._context)] for rule in self._by_kind.get(POSITION, ())],
                "move": [[rule.name, rule.weight(self._context)] for rule in self._by_kind.get(MOVE, ())],
            },
            indent=2,
        )

    def explain(self, state: State, player: str) -> tuple[tuple[RuleRecord, float], ...]:
        """Each position heuristic with what it adds to the player's value: its weight here times its reading."""
        return self._readings(self._by_kind.get(POSITION, ()), state, self._names(state, player))

    def _names(self, state: State, player: str) -> dict[str, object]:
        """What a heuristic reads besides the state's variables: `me`, `other`, `win_chance`, `wins`, `near`, `here`."""
        if self._consequence_library is None:
            return {"me": player}
        return self._consequence_library.names(self, state, player)

    def _weighed(self, rules: Sequence[RuleRecord], state: State, names: dict[str, object]) -> float | None:
        """The rules' readings, each times its weight in this context, summed; None where none could be read."""
        readings = self._readings(rules, state, names)
        return math.fsum(added for _, added in readings) if readings else None

    def _readings(
        self, rules: Sequence[RuleRecord], state: State, names: dict[str, object]
    ) -> tuple[tuple[RuleRecord, float], ...]:
        """Each rule with what it adds: its weight here times its reading. A rule reading nothing at this moment adds
        nothing; one that raises, or gives something other than a finite number, is left out."""
        added: list[tuple[RuleRecord, float]] = []
        for rule in rules:
            try:
                read = self._rule_caller.value(rule.rule, state, None, names, self._definitions(RULES_DEFINITIONS))
            except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                continue
            if read is None:
                added.append((rule, 0.0))
                continue
            if not isinstance(read, bool | int | float | np.bool_ | np.number):
                continue
            reading = float(read)  # type: ignore[arg-type]
            if math.isfinite(reading):
                added.append((rule, rule.weight(self._context) * reading))
        return tuple(added)

    def _of(self, *kinds: str) -> list[RuleRecord]:
        return [rule for kind in kinds for rule in self._by_kind.get(kind, ())]

    def _values(self, action: str) -> dict[str, object]:
        """Each parameter of the action with the rule giving its values, in the order they were declared."""
        return {
            str(rule.parameter): rule.rule
            for rule in sorted(self._by_kind.get(VALUES, ()), key=lambda rule: rule.id)
            if rule.action == action and rule.parameter
        }

    def _constraints(self, action: str) -> tuple[object, ...]:
        """The rules every legal action of that name satisfies, in the order they were declared."""
        return tuple(
            rule.rule for rule in sorted(self._by_kind.get(CONSTRAINT, ()), key=lambda rule: rule.id) if rule.action == action
        )

    def _effects(self, action: str) -> Effects:
        """What the action leads to: each effects rule with the chance it happens."""
        return tuple(
            (rule.probability, rule.rule)
            for rule in sorted(self._by_kind.get(EFFECTS, ()), key=lambda rule: rule.id)
            if rule.action == action
        )

    def _definitions(self, name: str) -> PythonRule | None:
        """The definitions script of that name, or the only one this context has."""
        rules = [rule for rule in self._by_kind.get(DEFINITIONS, ()) if isinstance(rule.rule, PythonRule)]
        for rule in rules:
            if rule.name == name:
                return rule.rule
        return rules[0].rule if len(rules) == 1 else None

    def _read(self, kind: str, what: str) -> object:
        """What a rule that reads nothing gives; a context without it isn't a game."""
        rules = self._by_kind.get(kind, ())
        if not rules:
            raise ValueError(f"{self._context} has no rule saying {what}")
        return self._rule_caller.value(rules[0].rule, NOTHING)

    def _call(self, kind: str, state: State, parameters: dict[str, object] | None = None) -> object:
        """What the rule of that kind says about the state, or None where the context has no such rule. A rule that
        raises gives None with a warning: it is the project's code, and the logs must survive it."""
        rules = self._by_kind.get(kind, ())
        if not rules:
            return None
        try:
            return self._rule_caller.value(rules[0].rule, state, parameters, None, self._definitions(EFFECTS_DEFINITIONS))
        except Exception:  # noqa: BLE001 - a game's rule is the project's code
            logger.warning("The %s %s rule raised", self._context, kind, exc_info=True)
            return None
