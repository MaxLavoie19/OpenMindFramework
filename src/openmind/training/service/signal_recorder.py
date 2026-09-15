import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np

from openmind.agent.model.domain import Domain
from openmind.inference.model.deduction import Deduction
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_readings import SignalReadings
from openmind.training.model.signal_record import SignalRecord
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: A position whose coming winner is known, with the winner's and the loser's names.
type Anchor = tuple[State, str, str]


class SignalRecorder:
    """Records how well signals point to the coming winner, in a two-player domain. Every position of the round's games
    whose coming winner is known is an anchor: one a deduction proved with payoffs that differ, the higher payoff's
    player winning, or any other position of a game whose payoffs differ. A proven draw is no anchor, whatever its game's
    result.

    Each signal with a source is read at every anchor for the winner and for the loser, in the term evaluator's workers:
    the winner reading higher agrees, the loser reading higher disagrees, a tie counts neither. A blank reading, the
    signal giving None, is a player lacking what it reads and counts as 0: blank for both is a tie. Winning reads the anchor's
    own result, so it always agrees; an aggregation votes with its parts, each part pointing to the player it reads higher,
    every part counting the same or by the weight it is given. A signal that can't be read on every anchor adds nothing
    this round. The readings add to the library's records, which keep counting across rounds; a position met in several
    games is an anchor in each."""

    def __init__(self, term_evaluator: TermEvaluator) -> None:
        self._term_evaluator = term_evaluator

    def record(
        self,
        domain: Domain,
        library: SignalLibrary,
        signals: Sequence[Signal],
        games: Sequence[PlayedGame],
        deductions: Sequence[Deduction] = (),
    ) -> SignalLibrary:
        """The library with every signal's record, new or added to: reads, then adds."""
        return self.add(library, signals, self.read(domain, signals, games, deductions))

    def read(
        self, domain: Domain, signals: Sequence[Signal], games: Sequence[PlayedGame], deductions: Sequence[Deduction] = ()
    ) -> SignalReadings:
        """Every signal's readings on the games' anchors, aggregations voting the same with their parts. A domain without
        exactly two players raises ValueError."""
        names = domain.players.names
        if len(names) != 2:
            raise ValueError(f"Signals are recorded between two players, not {len(names)}")
        anchors, proven = self._anchors(names, games, deductions)
        count = len(anchors)
        signs: dict[str, np.ndarray | None] = {}
        readable = [signal for signal in signals if signal.source is not None]
        if readable and anchors:
            rows = [PositionRow(state, winner, 0.0) for state, winner, _ in anchors] + [
                PositionRow(state, loser, 0.0) for state, _, loser in anchors
            ]
            columns = self._term_evaluator.columns(domain, rows, [signal.source for signal in readable])  # type: ignore[misc]
            for signal, column in zip(readable, columns, strict=True):
                if column is not None:
                    column = np.nan_to_num(column, nan=0.0)
                signs[signal.name] = None if column is None else np.sign(column[:count] - column[count:])
        for signal in readable:
            signs.setdefault(signal.name, np.zeros(count))
        for signal in signals:
            if signal.source is None and not signal.parts:
                signs[signal.name] = np.ones(count)
        readings = SignalReadings(
            count, proven, len(games), sum(1 for game in games if self._winner(game.payoffs) is not None), signs
        )
        for signal in signals:
            if signal.parts:
                readings = self.aggregate(readings, signal)
        return readings

    def aggregate(
        self, readings: SignalReadings, aggregation: Signal, weights: Mapping[str, float] | None = None
    ) -> SignalReadings:
        """The readings with the aggregation's: at every anchor, the sign of its readable parts' votes, each part weighing
        its weight, 1 without weights; None when no part could be read."""
        votes = [
            (weights or {}).get(part, 1.0) * readings.signs[part]  # type: ignore[operator]
            for part in aggregation.parts
            if readings.signs.get(part) is not None
        ]
        signs = dict(readings.signs)
        signs[aggregation.name] = np.sign(np.sum(votes, axis=0)) if votes else None
        return replace(readings, signs=signs)

    def add(self, library: SignalLibrary, signals: Sequence[Signal], readings: SignalReadings) -> SignalLibrary:
        """The library with the signals' readings added to their records."""
        records = {record.signal.name: record for record in library.records}
        unreadable = 0
        for signal in signals:
            record = records.get(signal.name, SignalRecord(signal))
            sign = readings.signs.get(signal.name)
            if sign is None:
                unreadable += 1
                records[signal.name] = replace(record, signal=signal)
                continue
            agreements, disagreements = int(np.count_nonzero(sign > 0)), int(np.count_nonzero(sign < 0))
            updated = replace(
                record,
                signal=signal,
                agreements=record.agreements + agreements,
                disagreements=record.disagreements + disagreements,
            )
            records[signal.name] = updated
            logger.debug(
                "Signal %s: %d agreements and %d disagreements this round, reliability %s",
                signal.name,
                agreements,
                disagreements,
                updated.reliability,
            )
        logger.info(
            "Recorded %d signals on %d anchors, %d of them proven, from %d decisive games of %d; %d signals couldn't be read",
            len(signals),
            readings.anchors,
            readings.proven,
            readings.decisive,
            readings.games,
            unreadable,
        )
        return SignalLibrary(library.domain, tuple(records.values()), library.supports)

    def _anchors(
        self, names: Sequence[str], games: Sequence[PlayedGame], deductions: Sequence[Deduction]
    ) -> tuple[list[Anchor], int]:
        """The anchors, in game order, and how many took their winner from a proof."""
        proven = {deduction.state: deduction.payoffs for deduction in deductions if deduction.payoffs is not None}
        anchors: list[Anchor] = []
        from_proofs = 0
        for game in games:
            result = self._winner(game.payoffs)
            for state in game.states:
                payoffs = proven.get(state)
                winner = result if payoffs is None else self._winner(payoffs)
                if winner is None:
                    continue
                from_proofs += payoffs is not None
                anchors.append((state, names[winner], names[1 - winner]))
        return anchors, from_proofs

    def _winner(self, payoffs: Sequence[float]) -> int | None:
        if payoffs[0] > payoffs[1]:
            return 0
        if payoffs[1] > payoffs[0]:
            return 1
        return None
