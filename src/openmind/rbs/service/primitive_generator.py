import itertools
from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.constant.consequence_constant import (
    ACTION,
    ME,
    NEAR,
    OTHER,
    OUTSIDE_NAME,
    SOLO_DISTANCE,
    WIN_CHANCE,
    WINS,
)
from openmind.rbs.constant.generation_constant import ANCHOR_PROBES, QUANTITY_CUTS
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.service.condition_evaluator import ConditionEvaluator
from openmind.rbs.service.consequence_library import ConsequenceLibrary
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.value import Value


class PrimitiveGenerator:
    """Generates the single conditions an action's hypotheses are built from, reading only the domain's players and the
    rows. A value that is a player's name is written relative to the player to act, as `me` or `other`. In order:

    1. every state variable at every value seen: `cell[2, 2] == me`; the variable naming the player to act stays
       absolute: `turn == 'X'`;
    2. every indexed variable read at the action's parameters, in every order: `cell[row, col] == None`;
    3. every parameter at every value seen: `col == 3`;
    4. near() at every offset up to max_offset from the variable the action sets, at the values that variable's base
       takes, me, other and OUTSIDE: `near(action, 0, -1) == me`;
    5. the conditions of the action's goal patterns;
    6. thresholds on quantities worked out from the domain's rules, at the values seen: win_chance(action), wins(me),
       wins(other), wins(me, action), wins(other, action), their changes, and solo_distance(me or other, action)."""

    def __init__(
        self,
        consequence_library: ConsequenceLibrary,
        condition_evaluator: ConditionEvaluator,
        state_namespace_mapper: StateNamespaceMapper,
        variable_name_mapper: VariableNameMapper,
    ) -> None:
        self._consequence_library = consequence_library
        self._condition_evaluator = condition_evaluator
        self._state_namespace_mapper = state_namespace_mapper
        self._variable_name_mapper = variable_name_mapper

    def primitives(
        self,
        domain: Domain,
        rows: Sequence[ActionRow],
        patterns: Sequence[GoalPattern],
        settings: GenerationSettings,
    ) -> tuple[PythonRule, ...]:
        players = set(domain.players.names)
        sources: dict[str, None] = {}

        values_by_variable: dict[str, dict[Value, None]] = {}
        values_by_parameter: dict[str, dict[Value, None]] = {}
        for row in rows:
            for name, value in row.state.variables:
                values_by_variable.setdefault(name, {})[value] = None
            for name, value in row.action.parameters:
                values_by_parameter.setdefault(name, {})[value] = None

        for name, values in values_by_variable.items():
            reading = self._state_namespace_mapper.to_source(name)
            for value in values:
                # The player to act compared with me is always true: which player it is has to stay absolute.
                renderings = (repr(value),) if name == domain.players.to_act else self._rendered(value, players)
                for rendered in renderings:
                    sources[f"{reading} == {rendered}"] = None

        parameters = list(values_by_parameter)
        values_by_base: dict[str, dict[Value, None]] = {}
        for name, values in values_by_variable.items():
            base, indices = self._variable_name_mapper.from_name(name)
            if indices and len(indices) == len(parameters):
                values_by_base.setdefault(base, {}).update(values)
        for base, values in values_by_base.items():
            for order in itertools.permutations(parameters):
                for value in values:
                    for rendered in self._rendered(value, players):
                        sources[f"{base}[{', '.join(order)}] == {rendered}"] = None

        for name, values in values_by_parameter.items():
            for value in values:
                sources[f"{name} == {value!r}"] = None

        anchors = [
            anchor
            for row in rows[:ANCHOR_PROBES]
            if (anchor := self._consequence_library.anchor(domain, row.state, row.action)) is not None
        ]
        if anchors:
            base, indices = anchors[0]
            renderings = dict.fromkeys((ME, OTHER, OUTSIDE_NAME))
            for name, values in values_by_variable.items():
                if self._variable_name_mapper.from_name(name)[0] == base:
                    renderings.update(dict.fromkeys(repr(value) for value in values if value not in players))
            steps = range(-settings.max_offset, settings.max_offset + 1)
            for offset in itertools.product(steps, repeat=len(indices)):
                if any(offset):
                    for rendered in renderings:
                        sources[f"{NEAR}({ACTION}, {', '.join(map(str, offset))}) == {rendered}"] = None

        for pattern in patterns:
            if rows and pattern.action == rows[0].action.name:
                sources.update(dict.fromkeys(condition.source for condition in pattern.conditions))

        quantities = self._quantities(settings)
        evaluated = self._condition_evaluator.all_values(domain, rows, [PythonRule(quantity) for quantity in quantities])
        for quantity, values in zip(quantities, evaluated, strict=True):
            if values is None:
                continue
            cuts = self._cuts([float(value) for value in values])  # type: ignore[arg-type]
            sources.update(dict.fromkeys(f"{quantity} >= {self._number(cut)}" for cut in cuts[1:]))
            sources.update(dict.fromkeys(f"{quantity} <= {self._number(cut)}" for cut in cuts[:-1]))

        return tuple(PythonRule(source) for source in sources)

    def _quantities(self, settings: GenerationSettings) -> list[str]:
        quantities = [
            f"{WIN_CHANCE}({ACTION})",
            f"{WINS}({ME})",
            f"{WINS}({OTHER})",
            f"{WINS}({ME}, {ACTION})",
            f"{WINS}({OTHER}, {ACTION})",
            f"{WINS}({ME}, {ACTION}) - {WINS}({ME})",
            f"{WINS}({OTHER}, {ACTION}) - {WINS}({OTHER})",
        ]
        if settings.solo_limit > 0:
            quantities.extend(
                f"{SOLO_DISTANCE}({player}, {ACTION}, {settings.solo_limit})" for player in (ME, OTHER)
            )
        return quantities

    def _rendered(self, value: Value, players: set[str]) -> tuple[str, ...]:
        return (ME, OTHER) if value in players else (repr(value),)

    def _cuts(self, values: list[float]) -> list[float]:
        distinct = sorted(set(values))
        if len(distinct) <= QUANTITY_CUTS:
            return distinct
        return sorted({float(cut) for cut in np.quantile(values, np.linspace(0.0, 1.0, QUANTITY_CUTS))})

    def _number(self, value: float) -> str:
        return str(int(value)) if float(value).is_integer() else repr(round(float(value), 6))
