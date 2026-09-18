import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from openmind.agent.constant.agent_constant import GAME_KEYWORD, LAST_ACTION
from openmind.agent.factory.game_factory import create_game
from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.service.game_memory import GameMemory
from openmind.dashboard.model.game_listing import GameListing
from openmind.dashboard.model.game_view import GameView
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.doxastic.model.record import Record
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.training.service.game_replayer import GameReplayer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

logger = logging.getLogger(__name__)


class GameBrowser:
    """Browses the decisive games a domain's knowledge base remembers, those whose payoffs differ: lists them, newest
    first, without replaying any, and shows one, replayed from its moves and drawn position by position with the domain's
    picture rule, or laid out as text without one. The game last drawn is kept, so a page reloading doesn't draw it
    again."""

    def __init__(self, game_factory: Callable[[str, object], RuleBasedSystem] = create_game) -> None:
        self._game_factory = game_factory
        self._games: dict[str, RuleBasedSystem] = {}
        self._kept: GameView | None = None

    def decisive(self, directory: Path, domain_name: str) -> tuple[GameListing, ...]:
        """Every decisive game, the newest first; none without a knowledge base."""
        knowledge_base = self._knowledge_base(directory, domain_name)
        return () if knowledge_base is None else tuple(reversed([self._listing(record) for record in self._records(knowledge_base)]))

    def latest(self, directory: Path, domain_name: str) -> GameView | None:
        """The newest decisive game, drawn; None without one."""
        listings = self.decisive(directory, domain_name)
        return None if not listings else self.game(directory, domain_name, listings[0].id)

    def game(self, directory: Path, domain_name: str, record_id: str) -> GameView | None:
        """The decisive game remembered under that record id, drawn; None when there's no such decisive game."""
        knowledge_base = self._knowledge_base(directory, domain_name)
        if knowledge_base is None:
            return None
        records = self._records(knowledge_base)
        ids = [record.id for record in records]
        if record_id not in ids:
            return None
        at = ids.index(record_id)
        previous_id = ids[at - 1] if at > 0 else None
        next_id = ids[at + 1] if at + 1 < len(ids) else None
        if self._kept is not None and self._kept.id == record_id:
            if (self._kept.previous_id, self._kept.next_id) != (previous_id, next_id):
                self._kept = replace(self._kept, previous_id=previous_id, next_id=next_id)
            return self._kept
        record = records[at]
        summary = GameSummaryJsonMapper().from_json(record.text, GameMemory(knowledge_base).models())
        rbs = self._game(domain_name, knowledge_base)
        positions = GameReplayer().positions(rbs, summary)
        actions = (None, *summary.actions)
        drawn = tuple(rbs.picture(state, **{LAST_ACTION: action}) for state, action in zip(positions, actions, strict=True))
        if all(picture is not None for picture in drawn):
            pictures = tuple(str(picture) for picture in drawn)
        else:
            text = GridTextMapper(VariableNameMapper())
            pictures = tuple(text.to_text(state) for state in positions)
        mapper = ActionTextMapper()
        listing = self._listing(record)
        self._kept = GameView(
            listing.label,
            listing.ended,
            listing.players,
            summary.payoffs,
            summary.ending,
            summary.record,
            tuple(mapper.to_text(action) for action in summary.actions),
            pictures,
            all(picture is not None for picture in drawn),
            record_id,
            previous_id,
            next_id,
        )
        logger.info("Drew %s: %d positions", summary.label, len(pictures))
        return self._kept

    def _knowledge_base(self, directory: Path, domain_name: str) -> KnowledgeBase | None:
        return create_knowledge_base(domain_name, directory) if (directory / domain_name).is_dir() else None

    def _records(self, knowledge_base: KnowledgeBase) -> list[Record]:
        """The decisive games' records, oldest first."""
        return [
            record
            for record in knowledge_base.recall(keyword=GAME_KEYWORD)
            if len({item["payoff"] for item in GameSummaryJsonMapper.players(record.text)}) > 1
        ]

    def _listing(self, record: Record) -> GameListing:
        data = GameSummaryJsonMapper.fields(record.text)
        players = data["players"]
        return GameListing(
            record.id,
            record.provenance.game or "",
            "" if record.provenance.when is None else record.provenance.when.replace(microsecond=0).isoformat(sep=" "),
            tuple((item["player"], item["model"]) for item in players),
            tuple(float(item["payoff"]) for item in players),
            data["ending"],
            int(data["plies"]),
        )

    def _game(self, name: str, knowledge_base: object) -> RuleBasedSystem:
        """The RBS for a game, kept once it has been built: its rules don't change while a page is browsed."""
        if name not in self._games:
            self._games[name] = self._game_factory(name, knowledge_base)
        return self._games[name]
