from datetime import datetime

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.context import Context
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.mechanism import Mechanism
from openmind.knowledge.model.opinion import Opinion
from openmind.knowledge.model.source import Source
from openmind.knowledge.model.tags import Tags
from openmind.knowledge.model.task import Task
from openmind.structure.model.value import Value


class KnowledgeJsonMapper:
    """Maps what the knowledge base keeps to JSON-ready dicts and back: direct experiences, beliefs, opinions, tasks,
    contexts and mechanisms. Values are kept as JSON keeps them; times as ISO text."""

    def source_to_data(self, source: Source) -> dict[str, object]:
        return {
            "mechanism": source.mechanism,
            "parameters": [[key, value] for key, value in source.parameters],
            "at": _time_to_data(source.at),
            "rests_on": list(source.rests_on),
        }

    def source_from_data(self, data: dict[str, object]) -> Source:
        return Source(
            str(data["mechanism"]),
            _pairs(data.get("parameters")),
            _time_from_data(data.get("at")),
            tuple(str(entry) for entry in data.get("rests_on", ())),  # type: ignore[union-attr]
        )

    def experience_to_data(self, experience: DirectExperience) -> dict[str, object]:
        return {
            "id": experience.id,
            "variable": experience.variable,
            "context": experience.context,
            "value": experience.value,
            "source": self.source_to_data(experience.source),
            "at": _time_to_data(experience.at),
            "tags": _pairs_to_data(experience.tags),
        }

    def experience_from_data(self, data: dict[str, object]) -> DirectExperience:
        return DirectExperience(
            str(data["variable"]),
            str(data["context"]),
            data.get("value"),  # type: ignore[arg-type]
            self.source_from_data(data["source"]),  # type: ignore[arg-type]
            _time_from_data(data.get("at")),
            _pairs(data.get("tags")),
            str(data.get("id", "")),
        )

    def belief_to_data(self, belief: Belief) -> dict[str, object]:
        return {
            "id": belief.id,
            "variable": belief.variable,
            "context": belief.context,
            "value": belief.value,
            "holder": list(belief.holder),
            "certainty": belief.certainty,
            "accuracy": belief.accuracy,
            "precision": belief.precision,
            "evidence": [
                {"value": evidence.value, "strength": evidence.strength, "source": self.source_to_data(evidence.source)}
                for evidence in belief.evidence
            ],
            "tags": _pairs_to_data(belief.tags),
        }

    def belief_from_data(self, data: dict[str, object]) -> Belief:
        return Belief(
            str(data["variable"]),
            str(data["context"]),
            data.get("value"),  # type: ignore[arg-type]
            tuple(str(name) for name in data.get("holder", ())),  # type: ignore[union-attr]
            float(data.get("certainty", 1.0)),  # type: ignore[arg-type]
            _float(data.get("accuracy")),
            _float(data.get("precision")),
            tuple(
                Evidence(item["value"], float(item["strength"]), self.source_from_data(item["source"]))
                for item in data.get("evidence", ())  # type: ignore[union-attr]
            ),
            _pairs(data.get("tags")),
            str(data.get("id", "")),
        )

    def opinion_to_data(self, opinion: Opinion) -> dict[str, object]:
        return {
            "id": opinion.id,
            "variable": opinion.variable,
            "context": opinion.context,
            "value": opinion.value,
            "at": _time_to_data(opinion.at),
            "reasons": [self.source_to_data(reason) for reason in opinion.reasons],
            "tags": _pairs_to_data(opinion.tags),
        }

    def opinion_from_data(self, data: dict[str, object]) -> Opinion:
        return Opinion(
            str(data["variable"]),
            str(data["context"]),
            data.get("value"),  # type: ignore[arg-type]
            _time_from_data(data.get("at")),
            tuple(self.source_from_data(reason) for reason in data.get("reasons", ())),  # type: ignore[union-attr]
            _pairs(data.get("tags")),
            str(data.get("id", "")),
        )

    def task_to_data(self, task: Task) -> dict[str, object]:
        return {
            "id": task.id,
            "name": task.name,
            "context": task.context,
            "value": [self.belief_to_data(belief) for belief in task.value],
            "expected_time": self.belief_to_data(task.expected_time),
            "status": task.status,
            "tags": _pairs_to_data(task.tags),
        }

    def task_from_data(self, data: dict[str, object]) -> Task:
        return Task(
            str(data["name"]),
            str(data["context"]),
            tuple(self.belief_from_data(item) for item in data.get("value", ())),  # type: ignore[union-attr]
            self.belief_from_data(data["expected_time"]),  # type: ignore[arg-type]
            str(data["status"]),
            _pairs(data.get("tags")),
            str(data.get("id", "")),
        )

    def context_to_data(self, context: Context) -> dict[str, object]:
        return {
            "id": context.id,
            "name": context.name,
            "parent": context.parent,
            "inherits": list(context.inherits),
            "tags": _pairs_to_data(context.tags),
        }

    def context_from_data(self, data: dict[str, object]) -> Context:
        parent = data.get("parent")
        return Context(
            str(data["id"]),
            str(data["name"]),
            None if parent is None else str(parent),
            tuple(str(context) for context in data.get("inherits", ())),  # type: ignore[union-attr]
            _pairs(data.get("tags")),
        )

    def mechanism_to_data(self, mechanism: Mechanism) -> dict[str, object]:
        return {
            "id": mechanism.id,
            "name": mechanism.name,
            "declared_accuracy": mechanism.declared_accuracy,
            "tags": _pairs_to_data(mechanism.tags),
        }

    def mechanism_from_data(self, data: dict[str, object]) -> Mechanism:
        return Mechanism(str(data["id"]), str(data["name"]), _float(data.get("declared_accuracy")), _pairs(data.get("tags")))

    def tags_to_data(self, tags: Tags) -> list[list[object]]:
        return _pairs_to_data(tags)

    def tags_from_data(self, data: object) -> Tags:
        return _pairs(data)


def _pairs_to_data(pairs: tuple[tuple[str, Value], ...]) -> list[list[object]]:
    return [[key, value] for key, value in pairs]


def _pairs(data: object) -> tuple[tuple[str, Value], ...]:
    return tuple((str(key), value) for key, value in (data or ()))  # type: ignore[union-attr, misc]


def _time_to_data(at: datetime | None) -> str | None:
    return None if at is None else at.isoformat()


def _time_from_data(data: object) -> datetime | None:
    return None if data is None else datetime.fromisoformat(str(data))


def _float(data: object) -> float | None:
    return None if data is None else float(data)  # type: ignore[arg-type]
