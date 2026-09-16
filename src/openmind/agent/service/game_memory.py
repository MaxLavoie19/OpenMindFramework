import logging
from collections.abc import Iterable

from openmind.agent.constant.agent_constant import DRAW, GAME_KEYWORD, LOSS, MODEL_KEYWORD, WIN
from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.doxastic.constant.doxastic_constant import PLAYED
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

#: What each outcome reads as in an outcome record.
OUTCOME_VERBS = {WIN: "won", DRAW: "drew", LOSS: "lost"}


class GameMemory:
    """Keeps finished games in the knowledge base, as each one ends:

    - each model that played, once, its text word for word, under its id and its name, with the keyword `model`;
    - the game, its summary as JSON, under the models' ids and names, with the keyword `game`;
    - what the game gave each player, `win`, `draw` or `loss` as the keyword, under that player's model's id and name.

    Every record is `played`, with the game and its round. An outcome is a fact, not evidence for a claim: a draw is a
    draw, and scores are counts of these records."""

    def __init__(self, knowledge_base: KnowledgeBase, mapper: GameSummaryJsonMapper | None = None) -> None:
        self._knowledge_base = knowledge_base
        self._mapper = GameSummaryJsonMapper() if mapper is None else mapper
        self._known: set[str] | None = None

    def remember(self, summary: GameSummary) -> None:
        """Writes the game's models not yet known, the game, and each player's outcome."""
        provenance = Provenance(PLAYED, game=summary.label, round=summary.round)
        subjects = (summary.domain,)
        for model in dict.fromkeys(summary.models):
            if model.id not in self._known_ids():
                self._knowledge_base.remember(
                    Record(model.text, provenance, subjects, (model.id, model.name), (MODEL_KEYWORD,))
                )
                self._known_ids().add(model.id)
                logger.info("Remembered model %s (%s)", model.id, model.name)
        names = tuple(dict.fromkeys(name for model in summary.models for name in (model.id, model.name)))
        self._knowledge_base.remember(
            Record(self._mapper.to_json(summary), provenance, subjects, names, (GAME_KEYWORD, summary.kind))
        )
        outcomes = self._outcomes(summary.payoffs)
        for player, model, outcome in zip(summary.players, summary.models, outcomes, strict=True):
            self._knowledge_base.remember(
                Record(
                    f"{model.name} {OUTCOME_VERBS[outcome]} as {player} in {summary.label}",
                    provenance,
                    subjects,
                    (model.id, model.name),
                    (outcome,),
                )
            )
        logger.debug(
            "Remembered %s: %s",
            summary.label,
            ", ".join(
                f"{model.name} {OUTCOME_VERBS[outcome]} as {player}"
                for player, model, outcome in zip(summary.players, summary.models, outcomes, strict=True)
            ),
        )

    def scores(self, names: Iterable[str]) -> dict[str, tuple[int, int, int, int]]:
        """Each name's games, wins, draws and losses, counting the outcome records under it: a model's name or id. A
        model playing both sides of a game counts both."""
        counts: dict[str, tuple[int, int, int, int]] = {}
        for name in names:
            wins, draws, losses = (len(self._knowledge_base.recall(name=name, keyword=outcome)) for outcome in (WIN, DRAW, LOSS))
            counts[name] = (wins + draws + losses, wins, draws, losses)
        return counts

    def models(self) -> tuple[ModelDescription, ...]:
        """Every model remembered, in the order they first played."""
        return tuple(
            ModelDescription(record.names[1], record.text)
            for record in self._knowledge_base.recall(keyword=MODEL_KEYWORD)
            if len(record.names) == 2
        )

    def _known_ids(self) -> set[str]:
        if self._known is None:
            self._known = {record.names[0] for record in self._knowledge_base.recall(keyword=MODEL_KEYWORD) if record.names}
        return self._known

    def _outcomes(self, payoffs: tuple[float, ...]) -> tuple[str, ...]:
        """Each player's outcome: the highest payoff alone wins, a highest payoff shared draws, anything lower loses."""
        best = max(payoffs)
        shared = payoffs.count(best) > 1
        return tuple((DRAW if shared else WIN) if payoff == best else LOSS for payoff in payoffs)
