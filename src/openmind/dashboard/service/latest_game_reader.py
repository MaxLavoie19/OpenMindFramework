import logging
from collections.abc import Callable
from pathlib import Path

from openmind.agent.constant.agent_constant import GAME_KEYWORD, LAST_ACTION
from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.model.domain import Domain
from openmind.agent.service.game_memory import GameMemory
from openmind.dashboard.model.game_view import GameView
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.training.service.game_replayer import GameReplayer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

logger = logging.getLogger(__name__)


class LatestGameReader:
    """Reads the latest decisive game a domain's knowledge base remembers, one whose payoffs differ, replayed from its
    moves and drawn position by position with the domain's picture rule, or laid out as text without one. The game last
    drawn is kept, so a page reloading doesn't draw it again until a newer decisive game is remembered."""

    def __init__(self, domain_factory: Callable[[str], Domain] = create_domain) -> None:
        self._domain_factory = domain_factory
        self._domains: dict[str, Domain] = {}
        self._kept: GameView | None = None

    def latest_decisive(self, directory: Path, domain_name: str) -> GameView | None:
        """None without a knowledge base or a decisive game."""
        if not (directory / domain_name).is_dir():
            return None
        knowledge_base = create_knowledge_base(domain_name, directory)
        records = [
            record
            for record in knowledge_base.recall(keyword=GAME_KEYWORD)
            if len(set(item["payoff"] for item in GameSummaryJsonMapper.players(record.text))) > 1
        ]
        if not records:
            return None
        latest = records[-1]
        when = "" if latest.provenance.when is None else latest.provenance.when.replace(microsecond=0).isoformat(sep=" ")
        if self._kept is not None and (self._kept.label, self._kept.ended) == (latest.provenance.game or "", when):
            return self._kept
        memory = GameMemory(knowledge_base)
        summary = GameSummaryJsonMapper().from_json(latest.text, memory.models())
        domain = self._domain(domain_name)
        positions = GameReplayer(create_predictor()).positions(domain, summary)
        actions = (None, *summary.actions)
        caller, text = create_rule_caller(), GridTextMapper(VariableNameMapper())
        if domain.picture is not None:
            pictures = tuple(
                str(caller.value(domain.picture, state, {LAST_ACTION: action}, None, domain.transitions.definitions))
                for state, action in zip(positions, actions, strict=True)
            )
        else:
            pictures = tuple(text.to_text(state) for state in positions)
        mapper = ActionTextMapper()
        self._kept = GameView(
            summary.label,
            when,
            tuple((player, model.name) for player, model in zip(summary.players, summary.models, strict=True)),
            summary.payoffs,
            summary.ending,
            summary.record,
            tuple(mapper.to_text(action) for action in summary.actions),
            pictures,
            domain.picture is not None,
        )
        logger.info("Drew %s: %d positions", summary.label, len(pictures))
        return self._kept

    def _domain(self, name: str) -> Domain:
        if name not in self._domains:
            self._domains[name] = self._domain_factory(name)
        return self._domains[name]
