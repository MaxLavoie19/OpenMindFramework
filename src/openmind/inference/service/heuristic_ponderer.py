import logging
import time
from collections.abc import Callable, Sequence

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.model.expression import Expression
from openmind.inference.model.ponder_settings import PonderSettings
from openmind.inference.model.pondering import Labelling, Pondering
from openmind.inference.service.expression_generator import ExpressionGenerator
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.inference.service.stability import Stability
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.factory.term_evaluator_factory import create_term_evaluator
from openmind.rbs.model.heuristic_target import HeuristicTarget
from openmind.rbs.model.move_row import MoveRow
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.service.game_relaxer import GameRelaxer
from openmind.rbs.service.term_evaluator import TermEvaluator
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.heuristic_finder import HeuristicFinder
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

#: What measuring how steady each term is is called. Like the reasoned worths it values no position and pays for
#: a search all the same, so it reports itself beside them or it is judged on nothing.
STEADY = "how steadily the terms of {context} read"


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

    **Nothing here is seeded with what a thing is worth, and nothing may be.** Asking the rules how much a thing
    can do and starting the search from that number is valuing a thing by what its rules admit and calling the
    answer a discovery. It read plausibly and it was measured resting on nothing — no position bore out that
    doing less is worse — so the ordering it produced was a theory wearing the clothes of a measurement. OMF is
    told to look for heuristics. That a piece has a value is a heuristic it finds in what games paid, or does not
    have."""

    def __init__(
        self,
        position_gatherer: PositionGatherer,
        position_deducer: PositionDeducer,
        heuristic_finder: HeuristicFinder,
        game_relaxer: GameRelaxer,
        expression_generator: ExpressionGenerator,
        term_evaluator: TermEvaluator | None = None,
        stability: Stability | None = None,
        telling: Callable[[Sequence[State]], Sequence[float | None]] | None = None,
    ) -> None:
        #: How to ask somebody who knows what each position is worth, or None where there is nobody to ask.
        #:
        #: **Handed over rather than held, because nothing here may know what a teller is.** A teller is a
        #: stronger player, a table, a person — whatever the domain has — and this sees a list of positions in
        #: and a list of numbers out. A run without one fits on what the games paid alone, exactly as before.
        self._telling = telling
        self._gatherer = position_gatherer
        self._deducer = position_deducer
        self._finder = heuristic_finder
        self._relaxer = game_relaxer
        self._expressions = expression_generator
        # What reads a term at a position, and what turns a walk of those readings into how steady each
        # term is. Both stateless, and given here so a caller may hand over its own.
        self._evaluator = create_term_evaluator() if term_evaluator is None else term_evaluator
        self._stability = Stability() if stability is None else stability

    def ponder(
        self,
        knowledge_base: KnowledgeBase,
        game: RuleBasedGame,
        settings: PonderSettings,
        positions: Sequence[State] = (),
    ) -> Pondering:
        """What it deduced, and what every way of deducing was worth.

        **Positions may be given, and where a game has been played they should be.** Gathered by walking, they
        are positions nothing ever settled: measured on chess, twenty gathered positions valued nought — none
        paid out because a random walk never reaches an ending, and none proved because a proof needs one
        within reach. Pondering then deduces nothing, however long it is given and however many relaxations it
        tries, and the fault is not in the search.

        A game that was played ends, and its last position paid somebody. Handed those, the payoff route has
        material for the first time — what the heuristic is being fitted to is what the game actually gave,
        which is the one thing here that is nobody's opinion.

        Walking stays the way to start on a game nobody has played yet, which is why it is still the default
        rather than a thing to remove."""
        started = time.monotonic()
        positions = tuple(positions) or self._gatherer.gather(
            game, settings.positions + settings.held_out, settings.seed
        )
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
        seeds, dropped = self._starting(game, training, settings, tried)
        generated = self._finder.generate(
            game,
            training,
            held_out,
            settings.values,
            HeuristicTarget(knowledge_base, game.context),
            seeds,
            dropped,
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
        along: list[tuple[State, tuple[float, ...]]] = []
        for state, payoffs in self._deducer.proven(seen):
            if state in settled or len(payoffs) != len(players):
                continue
            along.append((state, payoffs))
        # **The nodes first and the gathered positions last, because the last rows are the ones held back.**
        # `_split` keeps the tail to choose the price on, and these outnumber the gathered positions many
        # times over, so appending them would have quietly made the held-out set whatever the walks happened
        # to touch — endgames, where a proof is cheap — rather than the positions the heuristic is for. A price
        # chosen on a distribution the model will never meet is chosen on nothing.
        # **What a teller makes of each board, asked once for all of them.** Asking is dear to start and cheap
        # per position, so every position this ponder will fit on goes in one request — and a run with no
        # teller gets None everywhere, which is what having nobody to ask looks like.
        boards = [state for state, _ in (*along, *settled.items())]
        said = self._telling(boards) if self._telling is not None and boards else [None] * len(boards)
        for (state, payoffs), told in zip((*along, *settled.items()), said, strict=True):
            rows.extend(
                PositionRow(state, player, payoff, told)
                for player, payoff in zip(players, payoffs, strict=True)
            )
        labelling = Labelling(
            named, len(settled) + len(along), len({row.target for row in rows}), time.monotonic() - started
        )
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
            len(along),
            labelling.values,
            labelling.seconds,
        )
        return tuple(rows) if labelling.paid else ()

    def _starting(
        self,
        game: RuleBasedGame,
        training: Sequence[PositionRow],
        settings: PonderSettings,
        tried: list[Labelling],
    ) -> tuple[tuple[tuple[Expression, float] | Expression, ...], tuple[str, ...]]:
        """The terms worth trying first, and the templates of the terms not worth trying at all.

        **Only the positions that will be fitted on.** The held-out rows decide which price is kept, so a
        judgement reasoned partly out of them would make that choice partly a choice about rows it had already
        seen.

        **Nothing here proposes a term because of what the rules let a thing do.** What used to sit here asked
        the rules how much each thing could do and handed the search those terms already weighted, which is the
        one move OMF must not make: it is the answer being supplied and then found. What is left orders terms by
        how steadily each reads, which is a fact about the terms over the positions in hand and says nothing
        about what anything is worth."""
        positions = list({id(row.state): row.state for row in training}.values())
        vocabulary = self._expressions.vocabulary(game, positions)
        return self._steady(game, vocabulary, positions, settings, tried)

    def _steady(
        self,
        game: RuleBasedGame,
        vocabulary: object,
        positions: Sequence[State],
        settings: PonderSettings,
        tried: list[Labelling],
    ) -> tuple[tuple[Expression, ...], tuple[str, ...]]:
        """The leaves worth a generation, steadiest first, and the templates of those that are not.

        **Ordering, and dropping what the arithmetic says is nothing.** A term that varies a lot across a
        game and little between one position and the next is something to steer by; one that leaps about
        between neighbours says something nobody can act on, since a move cannot be chosen for its effect on
        a number that would have jumped anyway. Knowing that before the search spends a generation on it is
        the point.

        **Nothing is dropped unless `settings.steadiest` asks for it**, and that is measured rather than
        argued. Dropping the leaves whose columns never vary was defended twice — first as arithmetic, since a
        constant tells no position from another, then as arithmetic over the right sample once the spread was
        read from the fitted positions. Both defences were about a term's worth *on its own*, and the search's
        whole business is combining them: a leaf constant everywhere can be the half of a difference that is
        not. Dropping a twentieth of the candidates cost between two and thirteen times the held-out loss.

        What is kept is carried without a weight, because steadiness is not a claim about what a term is
        worth. A weight says where the fit should start; this says only where to look first.

        The spread is read over the positions being fitted, and the step between neighbours over the walks.
        They are facts about different things: what a term varies over is what the heuristic will meet, and a
        step exists only along a walk. Gathering skips a position it has seen and starts afresh when a game
        ends, so two of its positions side by side need not be a move apart — which is why the steps need
        walks — and a walk is a narrow slice of a game, which is why the spread must not come from one."""
        started = time.monotonic()
        leaves = self._expressions.leaves(vocabulary)  # type: ignore[arg-type]
        if not leaves or settings.walks < 1 or settings.walk_steps < 2:
            return (), ()
        runs = self._gatherer.walk(game, settings.walk_steps, settings.seed, settings.walks)
        read = tuple(self._read(game, run, leaves) for run in runs)
        # How much a term varies is a fact about the positions it will be fitted on, and how much it moves in
        # one step is a fact about steps, which only a walk has. Measuring both along the walk read a term
        # constant across nine opening positions as carrying nothing, when it varied freely over the eighty
        # being fitted — which is what made the first version of this worse than not having it.
        over = self._read(game, positions, leaves)
        # **Ordered always, dropped only where a budget asks.** Measured three ways over three seeds of
        # tic-tac-toe, held-out loss: untouched 0.0131, 0.0035, 0.0227; ordered and nothing dropped 0.0076,
        # 0.0035, 0.0227; ordered with the constant leaves dropped 0.0277, 0.0472, 0.0541. Ordering is free
        # and once better. Dropping costs 2.1, 13.5 and 2.4 times, and it costs that while removing only a
        # twentieth of the candidates — because a leaf that carries nothing by itself carries plenty as a
        # part, and dropping it takes everything the search would have grown from it. A column being constant
        # makes a term useless, not a building block useless.
        ordered, dropped = (
            (self._stability.steadiest(read, leaves, over), ())
            if settings.steadiest is None
            else self._stability.kept(read, leaves, settings.steadiest, over)
        )
        steady = self._stability.of(read, over)
        tried.append(
            Labelling(
                STEADY.format(context=game.context),
                sum(len(one) for one in runs),
                len({round(one, 9) for one in steady}),
                time.monotonic() - started,
            )
        )
        logger.info(
            "%s: %d terms over %d positions walked, %d told apart, %d kept and %d dropped, steadiest %s, "
            "in %.1f seconds",
            STEADY.format(context=game.context),
            len(leaves),
            sum(len(one) for one in runs),
            tried[-1].values,
            len(ordered),
            len(dropped),
            ordered[0].template if ordered else "none",
            tried[-1].seconds,
        )
        return ordered, tuple(one.template for one in dropped)  # type: ignore[return-value]

    def _read(self, game: RuleBasedGame, run: Sequence[State], leaves: Sequence[Expression]) -> tuple[tuple[float, ...], ...]:
        """What each term reads at each position of that walk, in the walk's order.

        A row per position with nobody's name and no target on it: what is wanted is the column, and the
        player and the payoff are what a row carries for a fit that is not happening here."""
        rows = [PositionRow(state, "", 0.0) for state in run]
        columns = self._evaluator.columns(game, rows, [self._expressions.source(one) for one in leaves])
        return tuple(
            tuple(0.0 if columns[term] is None else float(columns[term][at]) for term in range(len(leaves)))  # type: ignore[index]
            for at in range(len(rows))
        )

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
