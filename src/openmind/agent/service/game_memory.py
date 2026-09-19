import logging
from collections.abc import Iterable

from openmind.agent.constant.agent_constant import DRAW, GAME_KEYWORD, LOSS, MODEL_KEYWORD, WIN
from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.knowledge.constant.knowledge_constant import DIRECT_EXPERIENCE, SELF_PLAY
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: What each outcome reads as in the logs.
OUTCOME_VERBS = {WIN: "won", DRAW: "drew", LOSS: "lost"}


class GameMemory:
    """Keeps finished games in the knowledge base, as each one ends:

    - the game as a direct experience: its summary as JSON, word for word, with the keyword `game`, its kind, and every
      model that played it by id and name;
    - each model that played, once, as a belief holding its text, with the keyword `model`, its id and its name;
    - each player's payoff and outcome, `win`, `draw` or `loss`, as beliefs drawn from the game, tagged with the
      player's model's id and name, and the game's ending when it has one.

    An outcome is a fact drawn from the game, not an estimate: every belief here is held at 100 %, and scores are counts
    of the outcome beliefs."""

    def __init__(self, knowledge_base: KnowledgeBase, mapper: GameSummaryJsonMapper | None = None) -> None:
        self._knowledge_base = knowledge_base
        self._mapper = GameSummaryJsonMapper() if mapper is None else mapper

    def remember(self, summary: GameSummary) -> DirectExperience:
        """Keeps the game, the models not yet known, and what the game gave each player; gives back the game as the
        direct experience it is kept as."""
        context = self._knowledge_base.ensure_context(summary.domain).id
        for model in dict.fromkeys(summary.models):
            if self._knowledge_base.belief(f"model {model.id}", context) is None:
                self._knowledge_base.believe(
                    Belief(
                        f"model {model.id}",
                        context,
                        model.text,
                        tags=(("keyword", MODEL_KEYWORD), ("id", model.id), ("name", model.name)),
                    )
                )
                logger.info("Remembered model %s (%s)", model.id, model.name)
        game = self._knowledge_base.experience(
            DirectExperience(
                f"game {summary.label}",
                context,
                self._mapper.to_json(summary),
                Source(
                    self._knowledge_base.ensure_mechanism(SELF_PLAY).id,
                    (("game", summary.label), ("kind", summary.kind), ("round", summary.round)),
                ),
                tags=(
                    ("keyword", GAME_KEYWORD),
                    ("kind", summary.kind.casefold()),
                    *(("model", model.id) for model in dict.fromkeys(summary.models)),
                    *(("model", model.name) for model in dict.fromkeys(summary.models)),
                ),
            )
        )
        drawn_from = Source(self._knowledge_base.ensure_mechanism(DIRECT_EXPERIENCE).id, (), game.at, (game.id,))
        outcomes = self._outcomes(summary.payoffs)
        for player, model, payoff, outcome in zip(summary.players, summary.models, summary.payoffs, outcomes, strict=True):
            about = (("game", summary.label), ("player", player), ("model", model.id), ("model", model.name))
            self._drawn(f"payoff of {player} in {summary.label}", context, payoff, drawn_from, (("keyword", "payoff"), *about))
            self._drawn(f"outcome of {player} in {summary.label}", context, outcome, drawn_from, (("keyword", outcome), *about))
        if summary.ending is not None:
            self._drawn(
                f"ending of {summary.label}", context, summary.ending, drawn_from, (("keyword", "ending"), ("game", summary.label))
            )
        logger.debug(
            "Remembered %s as %s: %s",
            summary.label,
            game.id,
            ", ".join(
                f"{model.name} {OUTCOME_VERBS[outcome]} as {player}"
                for player, model, outcome in zip(summary.players, summary.models, outcomes, strict=True)
            ),
        )
        return game

    def scores(self, names: Iterable[str]) -> dict[str, tuple[int, int, int, int]]:
        """Each name's games, wins, draws and losses, counting the outcome beliefs tagged with it: a model's name or id.
        A model playing both sides of a game counts both."""
        counts: dict[str, tuple[int, int, int, int]] = {}
        for name in names:
            wins, draws, losses = (
                len(self._knowledge_base.beliefs(tags=(("keyword", outcome), ("model", name)))) for outcome in (WIN, DRAW, LOSS)
            )
            counts[name] = (wins + draws + losses, wins, draws, losses)
        return counts

    def games(self, kind: str | None = None) -> tuple[GameSummary, ...]:
        """Every game remembered, in the order they ended; those of one kind when it's named."""
        models = self.models()
        return tuple(self._mapper.from_json(str(experience.value), models) for experience in self.experiences(kind))

    def experiences(self, kind: str | None = None) -> tuple[DirectExperience, ...]:
        """Every game kept, as the direct experience it is, in the order they ended; those of one kind when it's named."""
        wanted = (("keyword", GAME_KEYWORD),) if kind is None else (("keyword", GAME_KEYWORD), ("kind", kind.casefold()))
        return self._knowledge_base.experiences(tags=wanted)

    def count(self, kind: str) -> int:
        """How many games of that kind are remembered."""
        return len(self.experiences(kind))

    def last_number(self, kind: str) -> int:
        """The highest number a remembered game of that kind has, 0 without one: numbering new games after it keeps every
        label its own, whichever games a stopped training managed to remember."""
        return max((game.number for game in self.games(kind)), default=0)

    def models(self) -> tuple[ModelDescription, ...]:
        """Every model remembered, in the order they first played."""
        return tuple(
            ModelDescription(str(dict(belief.tags)["name"]), str(belief.value))
            for belief in self._knowledge_base.beliefs(tags=(("keyword", MODEL_KEYWORD),))
        )

    def _drawn(self, variable: str, context: str, value: Value, source: Source, tags: tuple[tuple[str, Value], ...]) -> None:
        """A fact drawn from the game: believed at 100 %, with the game as its evidence."""
        self._knowledge_base.believe(
            Belief(variable, context, value, evidence=(Evidence(value, 1.0, source),), tags=tags)
        )

    def _outcomes(self, payoffs: tuple[float, ...]) -> tuple[str, ...]:
        """Each player's outcome: the highest payoff alone wins, a highest payoff shared draws, anything lower loses."""
        best = max(payoffs)
        shared = payoffs.count(best) > 1
        return tuple((DRAW if shared else WIN) if payoff == best else LOSS for payoff in payoffs)

