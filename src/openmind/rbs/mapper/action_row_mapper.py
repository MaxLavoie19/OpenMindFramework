from collections.abc import Sequence

import numpy as np

from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.model.action_row import ActionRow
from openmind.rbs.model.row_arrays import RowArrays
from openmind.world.model.action import Action
from openmind.world.model.state import State


class ActionRowMapper:
    """Maps search samples to action rows, and action rows to arrays. Samples of the same action in the same state are
    merged, visits summed and mean payoffs weighted by visits. Rows with fewer than min_visits visits are dropped; then
    each row's advantage is its mean payoff minus the best among its state's rows, and a state left with a single row is
    dropped, since it says nothing about which action to explore."""

    def to_rows(self, samples: Sequence[ActionSample], min_visits: int) -> tuple[ActionRow, ...]:
        totals: dict[tuple[State, Action], list[float]] = {}
        for sample in samples:
            total = totals.setdefault((sample.state, sample.action), [0.0, 0.0])
            total[0] += sample.visits
            total[1] += sample.visits * sample.mean_payoff
        by_state: dict[State, list[tuple[Action, int, float]]] = {}
        for (state, action), (visits, weighted) in totals.items():
            if visits > 0 and visits >= min_visits:
                by_state.setdefault(state, []).append((action, int(visits), weighted / visits))
        rows: list[ActionRow] = []
        for state, entries in by_state.items():
            if len(entries) < 2:
                continue
            best = max(mean for _, _, mean in entries)
            rows.extend(ActionRow(state, action, visits, mean, mean - best) for action, visits, mean in entries)
        return tuple(rows)

    def to_arrays(self, rows: Sequence[ActionRow]) -> RowArrays:
        index: dict[State, int] = {}
        return RowArrays(
            np.array([index.setdefault(row.state, len(index)) for row in rows], dtype=np.int64),
            np.array([row.visits for row in rows], dtype=float),
            np.array([row.advantage for row in rows], dtype=float),
            np.array([row.mean_payoff for row in rows], dtype=float),
        )
