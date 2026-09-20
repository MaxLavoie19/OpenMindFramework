import logging
import time
from collections.abc import Sequence

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.model.ponder_settings import PonderSettings
from openmind.inference.model.pondering import Labelling, Pondering
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.model.heuristic_target import HeuristicTarget
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: What valuing a position without playing is called, named with the game the values were reasoned out in: what that
#: game paid where it is over, and what its rules prove where they reach an end.
SETTLED = "what the rules of {context} settle"


class HeuristicPonderer:
    """Ponders a game before a single one is played: gathers positions by playing it, values what it can value
    without games, composes the candidates its readings allow, fits them, and keeps what held up.

    Nothing here knows any game. It has tools — the rules, the legal actions, a deduction, a relaxation, an expression
    search — and the deductions are its own. A game where holding pieces is worth nothing will not be told otherwise.

    Valuing a position without playing has two sources, and it tries them in this order:

    - **what the game paid**, where a gathered position is over, which is the only value known for certain;
    - **what a deduction proved**, where the rules prove a position within the plies it is given — in the game
      itself, and in each of its relaxations where the game itself taught too little. A relaxed game is a different
      game and what is proved there is a hint here, which is exactly what a bootstrap wants.

    What each source was worth is kept, paid or not: a way of bootstrapping that taught nothing here is a finding of
    its own."""

    def __init__(
        self,
        position_gatherer: PositionGatherer,
        position_deducer: PositionDeducer,
        value_generator: ValueGenerator,
        game_relaxer: GameRelaxer,
    ) -> None:
        self._gatherer = position_gatherer
        self._deducer = position_deducer
        self._generator = value_generator
        self._relaxer = game_relaxer

    def ponder(self, knowledge_base: KnowledgeBase, game: RuleBasedGame, settings: PonderSettings) -> Pondering:
        """What it deduced, and what every way of deducing was worth."""
        started = time.monotonic()
        positions = self._gatherer.gather(game, settings.positions + settings.held_out, settings.seed)
        tried: list[Labelling] = []
        rows = self._valued(game, game, positions, settings, tried)
        if not rows and settings.relaxations:
            for relaxation in self._relaxations(knowledge_base, game):
                rows = self._valued(game, relaxation, positions, settings, tried)
                if rows:
                    break
        if not rows:
            logger.info(
                "Pondered %s for %.1f seconds and deduced nothing: %s",
                game.context,
                time.monotonic() - started,
                "; ".join(f"{held.source} valued {held.positions}" for held in tried) or "nothing to value",
            )
            return Pondering(game.context, (), len(positions), tuple(tried), time.monotonic() - started)
        training, held_out = self._split(rows, settings.held_out)
        generated = self._generator.generate(
            game, training, held_out, settings.values, HeuristicTarget(knowledge_base, game.context)
        )
        logger.info(
            "Pondered %s for %.1f seconds: %d rules from %d positions valued by %s",
            game.context,
            time.monotonic() - started,
            len(generated.rules),
            len(training),
            tried[-1].source,
        )
        return Pondering(
            generated.context,
            generated.rules,
            len(positions),
            tuple(tried),
            time.monotonic() - started,
            generated.chosen.held_out_loss if generated.chosen is not None else None,
        )

    def _valued(
        self,
        game: RuleBasedGame,
        reasoned_in: RuleBasedGame,
        positions: Sequence[State],
        settings: PonderSettings,
        tried: list[Labelling],
    ) -> tuple[PositionRow, ...]:
        """The positions this game could settle, as rows: what it paid where a position is over, and what its rules
        prove where they reach an end within the plies given.

        The rows are of the real game, whichever game the values were reasoned out in: a relaxation is where the
        reasoning is cheap, not what the heuristic is for."""
        started = time.monotonic()
        named = SETTLED.format(context=reasoned_in.context)
        rows: list[PositionRow] = []
        valued, paid, proved = 0, 0, 0
        players = game.players().names
        for state in positions:
            payoffs = self._payoffs(game, state, players)
            if payoffs is not None:
                paid += 1
            else:
                payoffs = self._proved(reasoned_in, state, settings)
                proved += 1 if payoffs is not None else 0
            if payoffs is None:
                continue
            valued += 1
            rows.extend(PositionRow(state, player, payoff) for player, payoff in zip(players, payoffs, strict=True))
        labelling = Labelling(named, valued, len({row.target for row in rows}), time.monotonic() - started)
        tried.append(labelling)
        if valued == 0 and any(len(reasoned_in.joint_actions(state)) > 1 for state in positions[:1]):
            logger.warning(
                "%s settled nothing: several players act at once there, and a deduction reasons about one",
                named,
            )
        logger.info(
            "%s valued %d of %d positions — %d paid out, %d proved — %d differently, in %.1f seconds",
            named,
            valued,
            len(positions),
            paid,
            proved,
            labelling.values,
            labelling.seconds,
        )
        return tuple(rows) if labelling.paid else ()

    def _payoffs(self, game: RuleBasedGame, state: State, players: Sequence[str]) -> tuple[float, ...] | None:
        """What the position paid, where it is one the game is over in."""
        payoff = game.players().payoff
        held = state.model(payoff) if state.has(payoff) else None
        if held is None or not hasattr(held, "get"):
            return None
        values = [held.get(player) for player in players]
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
            return None
        return tuple(float(value) for value in values)  # type: ignore[arg-type]

    def _proved(self, game: RuleBasedGame, state: State, settings: PonderSettings) -> tuple[float, ...] | None:
        """What a deduction proves the position is worth, or None where the plies don't reach an end.

        A deduction reasons about one player acting. Where several can act at once — which is what a game relaxed of
        its turn rule is — it has nothing to say, and that is a limit of the tool, not of the position."""
        acting = game.joint_actions(state)
        if not acting or len(acting) > 1:
            return None
        deduction = self._deducer.deduce(game, state, DeductionBudget(settings.plies, settings.deduction_seconds))
        return deduction.payoffs

    def _relaxations(self, knowledge_base: KnowledgeBase, game: RuleBasedGame) -> tuple[RuleBasedGame, ...]:
        """Every relaxation the game's rules allow, as games of their own. A relaxation that can't be built is left
        out rather than stopping the pondering."""
        built: list[RuleBasedGame] = []
        for name in self._relaxer.relaxations(game.context):
            try:
                built.append(create_rule_based_game(knowledge_base, self._relaxer.relax(game.context, name)))
            except ValueError:
                logger.warning("Could not relax %s into %s", game.context, name, exc_info=True)
        return tuple(built)

    def _split(self, rows: Sequence[PositionRow], held_out: int) -> tuple[tuple[PositionRow, ...], tuple[PositionRow, ...]]:
        """The rows to fit on and the rows to choose on: the last positions gathered are held back, so a rule is
        chosen on positions it was never fitted on."""
        keep = min(held_out, max(len(rows) // 4, 0))
        return (tuple(rows[: len(rows) - keep]), tuple(rows[len(rows) - keep :])) if keep else (tuple(rows), ())
