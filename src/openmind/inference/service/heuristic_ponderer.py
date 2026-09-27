import logging
import time
from collections.abc import Sequence

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.model.expression import Expression
from openmind.inference.model.ponder_settings import PonderSettings
from openmind.inference.model.pondering import Labelling, Pondering
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.heuristic_deriver import HeuristicDeriver
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.inference.service.worth_reasoner import WorthReasoner
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.model.heuristic_target import HeuristicTarget
from openmind.rbs.model.move_row import MoveRow
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.value_generator import ValueGenerator
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: What valuing a position without playing is called, named with the game the values were reasoned out in: what that
#: game paid where it is over, and what its rules prove where they reach an end.
SETTLED = "what the rules of {context} settle"

#: What proving each action's payoff is called. The same deduction as `SETTLED` and a different question, so it
#: reports itself separately: a game whose positions can be settled but whose moves cannot is a game where one
#: of the two heuristics is worth fitting and the other is not, and nothing would say so if they shared a name.
MOVES_SETTLED = "what the rules of {context} settle a move pays"

#: What reasoning out the worth of things is called. It is not a way of valuing a position — it never looks at a
#: payoff — but it is a way of paying for a search, and a way of paying reports itself beside the others or it is
#: judged on nothing.
REASONED = "what the rules of {context} imply things are worth"


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
    its own.

    **And the rules say one thing more, which is not a value at all.** Asked what each thing on the board is
    worth — by taking it off and seeing how much of what its owner could do goes away — they answer without any
    position having been valued. That is not a target to fit against; it is a set of terms worth trying first,
    with the weight each ought to start at. It goes to the search as seeds. What it buys is expansion order: the
    term whose weight is what a knight is worth gets reached in the generation that would otherwise be spent
    rediscovering it, and a seed whose gradient does not pay is shrunk to nothing exactly like anything else."""

    def __init__(
        self,
        position_gatherer: PositionGatherer,
        position_deducer: PositionDeducer,
        value_generator: ValueGenerator,
        game_relaxer: GameRelaxer,
        heuristic_deriver: HeuristicDeriver,
        worth_reasoner: WorthReasoner,
        expression_generator: ExpressionGenerator,
    ) -> None:
        self._gatherer = position_gatherer
        self._deducer = position_deducer
        self._generator = value_generator
        self._relaxer = game_relaxer
        self._deriver = heuristic_deriver
        self._worth = worth_reasoner
        self._expressions = expression_generator

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
        valued_by = tried[-1].source
        seeds = self._seeded(game, training, settings, tried)
        generated = self._generator.generate(
            game, training, held_out, settings.values, HeuristicTarget(knowledge_base, game.context), seeds
        )
        logger.info(
            "Pondered %s for %.1f seconds: %d rules from %d positions valued by %s, %d terms seeded",
            game.context,
            time.monotonic() - started,
            len(generated.rules),
            len(training),
            valued_by,
            len(seeds),
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
        """The positions this game could settle, as rows: what it paid where a position is over, what its rules
        prove where they reach an end within the plies given, and every position those proofs passed through.

        The rows are of the real game, whichever game the values were reasoned out in: a relaxation is where the
        reasoning is cheap, not what the heuristic is for.

        **Every node the deduction proved, and not only the ones it was asked about.** Proving one position
        proves many — each state the walk reached and resolved is proven — and all of it but the answer was
        being thrown away. One memo now runs through every position of the pass, so a later position starts
        from what an earlier one already decided, and at the end the whole of it becomes rows.

        This is the one change the chess literature measures as large. Veness, Silver, Uther and Blair fit the
        same linear evaluation by the same self-play and got 1362 Elo labelling a search's root against 2157
        labelling its whole tree — the biggest single effect in that work, and a change to *what is labelled*
        rather than to what is fitted. Ours are proofs where theirs were backed-up estimates, which is a better
        label and a rarer one."""
        started = time.monotonic()
        named = SETTLED.format(context=reasoned_in.context)
        rows: list[PositionRow] = []
        valued, paid, proved = 0, 0, 0
        players = game.players().names
        settled: dict[State, tuple[float, ...]] = {}
        seen: dict = {}
        for state in positions:
            payoffs = self._payoffs(game, state, players)
            if payoffs is not None:
                paid += 1
            else:
                payoffs = self._proved(reasoned_in, state, settings, seen)
                proved += 1 if payoffs is not None else 0
            if payoffs is None:
                continue
            valued += 1
            settled[state] = payoffs
        # What the walks proved along the way, which is every node of every tree they built. A position asked
        # about outright keeps the value it was asked about, so nothing here overrides an answer with a note
        # taken on the way to it.
        along = 0
        for state, payoffs in self._deducer.proven(seen):
            if state in settled or len(payoffs) != len(players):
                continue
            settled[state] = payoffs
            along += 1
        for state, payoffs in settled.items():
            rows.extend(PositionRow(state, player, payoff) for player, payoff in zip(players, payoffs, strict=True))
        labelling = Labelling(named, len(settled), len({row.target for row in rows}), time.monotonic() - started)
        tried.append(labelling)
        if valued == 0 and any(len(reasoned_in.joint_actions(state)) > 1 for state in positions[:1]):
            logger.warning(
                "%s settled nothing: several players act at once there, and a deduction reasons about one",
                named,
            )
        logger.info(
            "%s valued %d of %d positions — %d paid out, %d proved — and %d more the proofs passed through, "
            "%d differently, in %.1f seconds",
            named,
            valued,
            len(positions),
            paid,
            proved,
            along,
            labelling.values,
            labelling.seconds,
        )
        return tuple(rows) if labelling.paid else ()

    def _seeded(
        self,
        game: RuleBasedGame,
        training: Sequence[PositionRow],
        settings: PonderSettings,
        tried: list[Labelling],
    ) -> tuple[tuple[Expression, float], ...]:
        """The terms the rules imply are worth trying, each with the weight they imply it should start at.

        **Only the positions that will be fitted on.** The held-out rows decide which price is kept, so a seed
        reasoned partly out of them would make that choice partly a choice about rows it had already seen. It
        costs nothing to avoid: what a thing is worth is a fact about the rules, and the fitted-on positions ask
        the rules just as well.

        **Nothing here is told whose anything is.** `HeuristicDeriver.seeds` takes deduced sides where there are
        any and this passes none, so the owning structure is found from the vocabulary instead — another base
        over the same places whose values are the players' names. That is `ExpressionGenerator.owning`, and for
        chess it finds the colour grid beside the piece grid without anything having deduced anything.
        `SideDeducer` is deliberately not wired in here: open question 33 has it concluding that one player owns
        both light and dark squares, and that white owns black's queen, and a deduction in that state would put
        wrong owners into the very condition the seed exists to carry. The seam is `seeds(..., sides=...)` for
        when the question is settled.

        **It reports itself whether or not it found anything.** A way of paying for a search that seeded nothing
        is a finding, exactly as a way of valuing that valued nothing is."""
        started = time.monotonic()
        positions = list({id(row.state): row.state for row in training}.values())
        vocabulary = self._expressions.vocabulary(game, positions)
        worth = self._worth.reason(game, positions, most=settings.worth_positions)
        seeds = self._deriver.seeds(self._deriver.holdings(worth), vocabulary)
        tried.append(
            Labelling(
                REASONED.format(context=game.context),
                len(positions) if settings.worth_positions is None else min(len(positions), settings.worth_positions),
                len({held for _, _, held in worth.holdings}),
                time.monotonic() - started,
            )
        )
        logger.info(
            "%s: %d things worth something over %d kinds, seeding %d terms, in %.1f seconds%s",
            REASONED.format(context=game.context),
            len(worth.holdings),
            len({value for _, value, _ in worth.holdings}),
            len(seeds),
            tried[-1].seconds,
            "" if worth.settled else " — resting on nothing, since no position bore out that doing less is worse",
        )
        return seeds

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

    def moved(
        self,
        game: RuleBasedGame,
        positions: Sequence[State],
        settings: PonderSettings,
        tried: list[Labelling],
    ) -> tuple[MoveRow, ...]:
        """What the rules prove each action in those positions pays the player taking it.

        **The other half of `_valued`, and the same deduction.** A position is valued by what the rules prove
        it is worth; an action is valued by what the rules prove it pays, which the same walk already worked
        out on the way to picking the best one and then threw away.

        **One source and not two, which is a correction to the plan this came from.** That plan had a ladder
        whose first rung was a successor being finished — worth what the game paid there — and whose second was
        the rules proving it. They are the same rung: a deduction of one ply calls the decision at depth zero,
        which finds no action and gives back the payoffs, so a finished successor is precisely what one ply
        proves. Two names for one walk would have been two rungs reporting the same evidence twice.

        Actions nothing could prove get no row, as positions nothing could settle get none: not knowing what a
        move pays is not the same as the move paying nothing, and a row saying zero is a claim the evidence
        never made.

        It reports itself as a `Labelling` like every other way of valuing, so a way that settles nothing says
        so rather than being silently absent."""
        started = time.monotonic()
        named = MOVES_SETTLED.format(context=game.context)
        rows: list[MoveRow] = []
        asked, answered = 0, 0
        for state in positions:
            acting = game.joint_actions(state)
            if not acting or len(acting) > 1:
                continue
            asked += 1
            player = game.acting_player(state)
            rated = self._deducer.deduce_moves(
                game, state, DeductionBudget(settings.plies, settings.deduction_seconds)
            )
            if not rated:
                continue
            answered += 1
            at = game.players().names.index(player)
            rows.extend(MoveRow(state, action, player, payoffs[at]) for action, payoffs in rated)
        labelling = Labelling(named, answered, len({row.target for row in rows}), time.monotonic() - started)
        tried.append(labelling)
        logger.info(
            "%s valued the moves of %d of %d positions it could ask about, %d actions in all, %d differently, "
            "in %.1f seconds",
            named,
            answered,
            asked,
            len(rows),
            labelling.values,
            labelling.seconds,
        )
        return tuple(rows) if labelling.paid else ()

    def _proved(
        self, game: RuleBasedGame, state: State, settings: PonderSettings, seen: dict | None = None
    ) -> tuple[float, ...] | None:
        """What a deduction proves the position is worth, or None where the plies don't reach an end.

        A deduction reasons about one player acting. Where several can act at once — which is what a game relaxed of
        its turn rule is — it has nothing to say, and that is a limit of the tool, not of the position."""
        acting = game.joint_actions(state)
        if not acting or len(acting) > 1:
            return None
        deduction = self._deducer.deduce(
            game, state, DeductionBudget(settings.plies, settings.deduction_seconds), seen
        )
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
