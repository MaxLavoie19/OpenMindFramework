from collections.abc import Mapping
from dataclasses import dataclass, field

from openmind.knowledge.model.model_record import ModelRecord
from openmind.search.model.search_settings import SearchSettings


@dataclass(frozen=True, slots=True)
class Allocation:
    """What the time management policy decided for one step: which model each task runs with, and how much the planner
    may explore.

    A task with no model is left out, and whatever needed it does without."""

    settings: SearchSettings
    models: Mapping[str, ModelRecord] = field(default_factory=dict)

    def of(self, task: str) -> ModelRecord | None:
        return self.models.get(task)
