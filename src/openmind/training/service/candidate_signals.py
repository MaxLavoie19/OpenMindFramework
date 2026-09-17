import logging
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.inference.model.expression import Expression
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary

logger = logging.getLogger(__name__)


class CandidateSignals:
    """The signals read from positions: the signals already recorded, as they are, such as those deduced from the rules
    with their names and premises; then, each named by its source, the seeds proofs induced, the rules the library's
    signals support, and the leaves of the games' positions; once each by source."""

    def __init__(self, expression_generator: ExpressionGenerator) -> None:
        self._expression_generator = expression_generator

    def candidates(
        self, domain: Domain, games: Sequence[PlayedGame], seeds: Sequence[Expression], library: SignalLibrary
    ) -> tuple[Signal, ...]:
        generator = self._expression_generator
        vocabulary = generator.vocabulary(domain, (state for game in games for state in game.states))
        recorded = [record.signal for record in library.records if record.signal.source is not None]
        known = {signal.source for signal in recorded}
        sources = [
            *(generator.source(seed) for seed in seeds),
            *(support.term for support in library.supports),
            *(generator.source(leaf) for leaf in generator.leaves(vocabulary)),
        ]
        candidates = (*recorded, *(Signal(source.source, source) for source in dict.fromkeys(sources) if source not in known))
        logger.info(
            "%d candidate signals: %d seeds, %d supported rules, %d recorded signals, and the leaves of %d games' positions",
            len(candidates),
            len(seeds),
            len(library.supports),
            len(recorded),
            len(games),
        )
        return candidates
