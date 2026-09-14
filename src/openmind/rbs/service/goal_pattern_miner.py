import logging
from collections import Counter
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.rbs.constant.consequence_constant import ACTION, ME, NEAR, OTHER
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)


class GoalPatternMiner:
    """Finds what winning moves need, for any domain. For each distinct sampled move that wins with certainty, every
    variable holding a player's name is changed, one at a time, to a value that isn't a player's name, first one seen
    for that variable, else for its base. A variable whose change makes the move illegal or no longer a certain win is
    needed. The needed variables form the move's goal pattern, written relative to the mover (me, other) and, where
    they share the base and index count of the variable the move sets, as offsets from it (near)."""

    def __init__(
        self,
        consequence_library: ConsequenceLibrary,
        solver: Solver,
        state_namespace_mapper: StateNamespaceMapper,
        variable_name_mapper: VariableNameMapper,
    ) -> None:
        self._consequence_library = consequence_library
        self._solver = solver
        self._state_namespace_mapper = state_namespace_mapper
        self._variable_name_mapper = variable_name_mapper

    def patterns(self, domain: Domain, rows: Sequence[ActionRow], limit: int) -> tuple[GoalPattern, ...]:
        """The goal patterns of up to `limit` winning moves among the rows, the most frequent first."""
        players = set(domain.players.names)
        alternatives = self._alternatives(rows, players)
        winning = []
        for state, action in dict.fromkeys((row.state, row.action) for row in rows):
            if len(winning) >= limit:
                break
            if self._consequence_library.win_chance(domain, state, action) == 1:
                winning.append((state, action))

        counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
        for state, action in winning:
            conditions = self._needed(domain, state, action, players, alternatives)
            if conditions:
                counts[(action.name, conditions)] += 1
        patterns = tuple(
            GoalPattern(name, tuple(PythonRule(source) for source in conditions), moves)
            for (name, conditions), moves in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
        logger.info("Found %d goal patterns in %d winning moves", len(patterns), len(winning))
        if logger.isEnabledFor(logging.DEBUG):
            for pattern in patterns:
                logger.debug(
                    "Goal pattern of %s in %d moves: %s",
                    pattern.action,
                    pattern.moves,
                    " and ".join(condition.source for condition in pattern.conditions),
                )
        return patterns

    def _needed(
        self,
        domain: Domain,
        state: State,
        action: object,
        players: set[str],
        alternatives: dict[str, Value],
    ) -> tuple[str, ...]:
        library = self._consequence_library
        mover = library.names(domain, state)[ME]
        other = library.names(domain, state)[OTHER]
        anchor = library.anchor(domain, state, action)  # type: ignore[arg-type]
        needed: list[str] = []
        for index, (name, value) in enumerate(state.variables):
            if value not in players or name == domain.players.to_act:
                continue
            base, _ = self._variable_name_mapper.from_name(name)
            alternative = alternatives.get(name, alternatives.get(base, _MISSING))
            if alternative is _MISSING:
                continue
            changed = State((*state.variables[:index], (name, alternative), *state.variables[index + 1 :]))
            if action in self._solver.solve(domain.problem, changed) and library.win_chance(domain, changed, action) == 1:  # type: ignore[arg-type]
                continue
            relative = ME if value == mover else OTHER if value == other else repr(value)
            needed.append(f"{self._reading(name, anchor)} == {relative}")
        return tuple(sorted(needed))

    def _reading(self, name: str, anchor: tuple[str, tuple[int, ...]] | None) -> str:
        base, texts = self._variable_name_mapper.from_name(name)
        if (
            anchor is not None
            and base == anchor[0]
            and len(texts) == len(anchor[1])
            and all(text.lstrip("-").isdecimal() for text in texts)
        ):
            offset = ", ".join(str(int(text) - index) for text, index in zip(texts, anchor[1]))
            return f"{NEAR}({ACTION}, {offset})"
        return self._state_namespace_mapper.to_source(name)

    def _alternatives(self, rows: Sequence[ActionRow], players: set[str]) -> dict[str, Value]:
        """For each variable and each base, the first value seen that isn't a player's name."""
        alternatives: dict[str, Value] = {}
        for row in rows:
            for name, value in row.state.variables:
                if value in players:
                    continue
                alternatives.setdefault(name, value)
                alternatives.setdefault(self._variable_name_mapper.from_name(name)[0], value)
        return alternatives


_MISSING = object()
