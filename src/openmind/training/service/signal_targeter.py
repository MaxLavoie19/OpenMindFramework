import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import expit

from openmind.agent.model.domain import Domain
from openmind.inference.model.deduction import Deduction
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.training.constant.signal_constant import UNIFORM, WEIGHTED, WIN
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.signal import Signal
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class SignalTargeter:
    """Values every position of a round's games, once for each player, at the targets of the signals the round follows,
    in a two-player domain:

    - a signal with a source is read `horizon` plies later in the same game, or at its last position when the game ends
      sooner, for the player and for the other player; its target is expit of that difference standardized over the
      round's rows, so 0.5 when both read the same;
    - winning's target is the player's final payoff;
    - `uniform` is expit of the mean of the standardized differences of every signal followed, winning's being its
      standardized payoff difference; `weighted` weighs each by its reliability, and is uniform while every weight is 0;
    - a position a deduction proved takes the proven payoffs in every target: a proof is exact where signals are
      heuristics.

    A blank reading, the signal giving None, is a player lacking what the signal reads and counts as 0 in the difference.
    Blank for both players is no reading: the signal's target there is 0.5, its spread is taken over the rows with a
    reading, and the aggregations average only the signals with a reading at each row. A signal that can't be read on
    every row gets no target this round."""

    def __init__(self, term_evaluator: TermEvaluator) -> None:
        self._term_evaluator = term_evaluator

    def rows(self, domain: Domain, games: Sequence[PlayedGame]) -> tuple[PositionRow, ...]:
        """Every position of every game, for each player in the players' order, valued at the player's final payoff."""
        names = domain.players.names
        return tuple(
            PositionRow(state, name, payoff)
            for game in games
            for state in game.states
            for name, payoff in zip(names, game.payoffs, strict=True)
        )

    def targets(
        self,
        domain: Domain,
        games: Sequence[PlayedGame],
        signals: Sequence[Signal],
        reliabilities: Mapping[str, float],
        horizon: int,
        deductions: Sequence[Deduction] = (),
    ) -> dict[str, np.ndarray]:
        """By signal name, in the signals' order, the targets of `rows(domain, games)`. A domain without exactly two
        players raises ValueError."""
        names = domain.players.names
        if len(names) != 2:
            raise ValueError(f"Signal targets are read between two players, not {len(names)}")
        read_rows: list[PositionRow] = []
        payoffs: list[float] = []
        proven_payoffs: list[tuple[float, ...] | None] = []
        proofs = {deduction.state: deduction.payoffs for deduction in deductions if deduction.payoffs is not None}
        for game in games:
            last = len(game.states) - 1
            for at, state in enumerate(game.states):
                read = game.states[min(at + horizon, last)]
                for index, name in enumerate(names):
                    read_rows.append(PositionRow(read, name, 0.0))
                    payoffs.append(game.payoffs[index])
                    proven = proofs.get(state)
                    proven_payoffs.append(None if proven is None else (proven[index],))
        count = len(read_rows)
        standardized: dict[str, np.ndarray] = {}
        present: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        readable = [signal for signal in signals if signal.source is not None]
        columns = self._term_evaluator.columns(domain, read_rows, [signal.source for signal in readable]) if readable and count else []  # type: ignore[misc]
        for signal, column in zip(readable, columns, strict=True):
            if column is None:
                logger.info("Signal %s couldn't be read on every row: no target this round", signal.name)
                continue
            present[signal.name] = self._present(column)
            standardized[signal.name] = self._standardized(self._differences(np.nan_to_num(column, nan=0.0)), present[signal.name])
        payoff_column = np.array(payoffs, dtype=float)
        for signal in signals:
            if signal.source is None and not signal.parts:
                present[signal.name] = np.ones(count, dtype=bool)
                standardized[signal.name] = self._standardized(self._differences(payoff_column), present[signal.name])
        followed = [name for name in standardized]
        for signal in signals:
            if signal.name in standardized and signal.source is not None:
                targets[signal.name] = expit(standardized[signal.name])
            elif signal.source is None and not signal.parts:
                targets[signal.name] = payoff_column.copy()
            elif signal.name == UNIFORM:
                targets[signal.name] = expit(self._combined(standardized, present, followed, None))
            elif signal.name == WEIGHTED:
                targets[signal.name] = expit(self._combined(standardized, present, followed, reliabilities))
        for at, proven in enumerate(proven_payoffs):
            if proven is not None:
                for values in targets.values():
                    values[at] = proven[0]
        logger.info(
            "Targets for %d signals on %d rows, read %d plies later; %d rows proven",
            len(targets),
            count,
            horizon,
            sum(1 for proven in proven_payoffs if proven is not None),
        )
        return {signal.name: targets[signal.name] for signal in signals if signal.name in targets}

    def _differences(self, column: np.ndarray) -> np.ndarray:
        """Each row's reading minus the other player's at the same position, rows coming in pairs."""
        first, second = column[0::2], column[1::2]
        differences = np.empty_like(column, dtype=float)
        differences[0::2], differences[1::2] = first - second, second - first
        return differences

    def _present(self, column: np.ndarray) -> np.ndarray:
        """Whether each row has a reading: the player's or the other player's reading isn't blank."""
        has = ~np.isnan(column)
        either = has[0::2] | has[1::2]
        present = np.empty(len(column), dtype=bool)
        present[0::2], present[1::2] = either, either
        return present

    def _standardized(self, differences: np.ndarray, present: np.ndarray) -> np.ndarray:
        """The differences divided by their spread over the rows with a reading; 0 on the others."""
        read = differences[present]
        scale = float(np.sqrt(np.mean(read**2))) if len(read) else 0.0
        if scale == 0.0 or not math.isfinite(scale):
            return np.zeros_like(differences)
        return np.where(present, differences / scale, 0.0)

    def _combined(
        self,
        standardized: Mapping[str, np.ndarray],
        present: Mapping[str, np.ndarray],
        names: Sequence[str],
        weights: Mapping[str, float] | None,
    ) -> np.ndarray:
        """At each row, the weighted mean of the standardized differences of the signals with a reading there; 0 where
        none has one."""
        if not names:
            return np.zeros(0)
        given = [(weights or {}).get(name, 1.0) if weights is not None else 1.0 for name in names]
        if math.fsum(given) <= 0.0:
            given = [1.0] * len(names)
        total = sum(weight * present[name] for weight, name in zip(given, names, strict=True))
        summed = sum(weight * standardized[name] for weight, name in zip(given, names, strict=True))
        with np.errstate(all="ignore"):
            return np.where(total > 0.0, summed / np.where(total > 0.0, total, 1.0), 0.0)  # type: ignore[arg-type]
