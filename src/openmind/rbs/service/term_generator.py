import logging
from collections.abc import Sequence

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.rbs.constant.consequence_constant import ME, OTHER, SOLO_DISTANCE, WINS
from openmind.rbs.constant.value_constant import COUNT_VARIABLE, MAX_VALUE_TERMS
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.value import Value

logger = logging.getLogger(__name__)


class TermGenerator:
    """Generates the single terms a position's value is fitted from, reading only the domain's players and the rows. A
    value that is a player's name is written relative to the player valued, as `me` or `other`. In order:

    1. every state variable at every value seen: `cell[2, 2] == me`; the variable naming the player to act is also
       written absolutely: `turn == 'X'`;
    2. for every indexed variable, how many of its variables hold each value seen:
       `sum(value == me for value in cell.values())`;
    3. quantities worked out from the domain's rules, wins(me), wins(other) and, unless solo_limit is 0,
       solo_distance(me, None, solo_limit) and solo_distance(other, None, solo_limit), each as it is and at up to
       `cuts` thresholds seen: `wins(other) >= 1`.

    A variable with more than MAX_VALUE_TERMS distinct values, such as a history of positions or a move clock, gets no
    term per value, and an indexed variable's base with that many gets no count terms."""

    def __init__(
        self,
        term_evaluator: TermEvaluator,
        state_namespace_mapper: StateNamespaceMapper,
        variable_name_mapper: VariableNameMapper,
    ) -> None:
        self._term_evaluator = term_evaluator
        self._state_namespace_mapper = state_namespace_mapper
        self._variable_name_mapper = variable_name_mapper

    def generate(self, domain: Domain, rows: Sequence[PositionRow], settings: ValueSettings) -> tuple[PythonRule, ...]:
        players = set(domain.players.names)
        sources: dict[str, None] = {}

        values_by_variable: dict[str, dict[Value, None]] = {}
        for row in rows:
            for name, value in row.state.variables:
                values_by_variable.setdefault(name, {})[value] = None

        values_by_base: dict[str, dict[Value, None]] = {}
        left_out: list[str] = []
        for name, values in values_by_variable.items():
            if len(values) > MAX_VALUE_TERMS:
                left_out.append(name)
                continue
            reading = self._state_namespace_mapper.to_source(name)
            for value in values:
                renderings = self._rendered(value, players)
                if name == domain.players.to_act:
                    renderings = (repr(value), *renderings)
                for rendered in dict.fromkeys(renderings):
                    sources[f"{reading} == {rendered}"] = None
            base, indices = self._variable_name_mapper.from_name(name)
            if indices:
                values_by_base.setdefault(base, {}).update(values)

        for base, values in values_by_base.items():
            if len(values) > MAX_VALUE_TERMS:
                left_out.append(base)
                continue
            for value in values:
                for rendered in self._rendered(value, players):
                    sources[f"sum({COUNT_VARIABLE} == {rendered} for {COUNT_VARIABLE} in {base}.values())"] = None

        if left_out:
            logger.info(
                "Left out the values of %d variables with more than %d distinct values: %s",
                len(left_out),
                MAX_VALUE_TERMS,
                ", ".join(left_out),
            )

        quantities = [f"{WINS}({ME})", f"{WINS}({OTHER})"]
        if settings.solo_limit > 0:
            quantities.extend(f"{SOLO_DISTANCE}({player}, None, {settings.solo_limit})" for player in (ME, OTHER))
        columns = self._term_evaluator.columns(domain, rows, [PythonRule(quantity) for quantity in quantities])
        for quantity, column in zip(quantities, columns, strict=True):
            if column is None:
                continue
            sources[quantity] = None
            cuts = self._cuts(column, settings.cuts)
            sources.update(dict.fromkeys(f"{quantity} >= {self._number(cut)}" for cut in cuts[1:]))

        return tuple(PythonRule(source) for source in sources)

    def _rendered(self, value: Value, players: set[str]) -> tuple[str, ...]:
        return (ME, OTHER) if value in players else (repr(value),)

    def _cuts(self, column: np.ndarray, cuts: int) -> list[float]:
        distinct = sorted({float(value) for value in column})
        if len(distinct) <= cuts:
            return distinct
        return sorted({float(cut) for cut in np.quantile(column, np.linspace(0.0, 1.0, cuts))})

    def _number(self, value: float) -> str:
        return str(int(value)) if float(value).is_integer() else repr(round(float(value), 6))
