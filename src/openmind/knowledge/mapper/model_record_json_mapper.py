from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.model.model_record import ModelRecord


class ModelRecordJsonMapper:
    """Maps a model record to a JSON object and back."""

    def __init__(self, knowledge_json_mapper: KnowledgeJsonMapper | None = None) -> None:
        self._knowledge = KnowledgeJsonMapper() if knowledge_json_mapper is None else knowledge_json_mapper

    def to_data(self, model: ModelRecord) -> dict[str, object]:
        return {
            "id": model.id,
            "name": model.name,
            "task": model.task,
            "context": model.context,
            "family": model.family,
            "mechanism": model.mechanism,
            "location": model.location,
            "tags": self._knowledge.tags_to_data(model.tags),
        }

    def from_data(self, data: dict[str, object]) -> ModelRecord:
        return ModelRecord(
            str(data["name"]),
            str(data["task"]),
            str(data["context"]),
            str(data["family"]),
            str(data["mechanism"]),
            str(data.get("location", "")),
            self._knowledge.tags_from_data(data.get("tags")),
            str(data.get("id", "")),
        )
