from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.model.ruleset import Ruleset
from openmind.knowledge.model.ruleset_link import RulesetLink


class RulesetJsonMapper:
    """Maps a ruleset to a JSON object and back: its rules by id, each with its weight there."""

    def __init__(self, knowledge_json_mapper: KnowledgeJsonMapper | None = None) -> None:
        self._knowledge = KnowledgeJsonMapper() if knowledge_json_mapper is None else knowledge_json_mapper

    def to_data(self, ruleset: Ruleset) -> dict[str, object]:
        return {
            "id": ruleset.id,
            "name": ruleset.name,
            "context": ruleset.context,
            "task": ruleset.task,
            "source": self._knowledge.source_to_data(ruleset.source),
            "links": [[link.rule, link.weight] for link in ruleset.links],
            "open": ruleset.open,
            "tags": self._knowledge.tags_to_data(ruleset.tags),
        }

    def from_data(self, data: dict[str, object]) -> Ruleset:
        return Ruleset(
            str(data["name"]),
            str(data["context"]),
            str(data["task"]),
            self._knowledge.source_from_data(data["source"]),  # type: ignore[arg-type]
            tuple(RulesetLink(str(rule), float(weight)) for rule, weight in data.get("links", ())),  # type: ignore[misc]
            bool(data.get("open", False)),
            self._knowledge.tags_from_data(data.get("tags")),
            str(data.get("id", "")),
        )
