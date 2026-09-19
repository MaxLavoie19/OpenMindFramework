from dataclasses import dataclass

from openmind.knowledge.constant.knowledge_constant import PENDING
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Task:
    """Something OMF can spend time on, in a context. `value` holds beliefs of its worth over several measures — utility
    gained, precision gained, time saved — weighed together by preferences; they drift as runs of the task report what
    they brought. `expected_time` is a belief too, in seconds. `status` is pending, running or done. `id` is empty until
    the knowledge base keeps it."""

    name: str
    context: str
    value: tuple[Belief, ...]
    expected_time: Belief
    status: str = PENDING
    tags: Tags = ()
    id: str = ""
