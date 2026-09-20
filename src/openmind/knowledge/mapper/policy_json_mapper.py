from openmind.knowledge.mapper.knowledge_json_mapper import KnowledgeJsonMapper
from openmind.knowledge.model.goal import Goal
from openmind.knowledge.model.policy import Policy
from openmind.knowledge.model.preference import Preference


class PolicyJsonMapper:
    """Maps a policy, a goal and a preference to JSON objects and back."""

    def __init__(self, knowledge_json_mapper: KnowledgeJsonMapper | None = None) -> None:
        self._knowledge = KnowledgeJsonMapper() if knowledge_json_mapper is None else knowledge_json_mapper

    def policy_to_data(self, policy: Policy) -> dict[str, object]:
        return {
            "id": policy.id,
            "name": policy.name,
            "context": policy.context,
            "sub_goal": policy.sub_goal,
            "position_value": policy.position_value,
            "move_value": policy.move_value,
            "tags": self._knowledge.tags_to_data(policy.tags),
        }

    def policy_from_data(self, data: dict[str, object]) -> Policy:
        return Policy(
            str(data["name"]),
            str(data["context"]),
            str(data.get("sub_goal", "")),
            str(data.get("position_value", "")),
            str(data.get("move_value", "")),
            self._knowledge.tags_from_data(data.get("tags")),
            str(data.get("id", "")),
        )

    def goal_to_data(self, goal: Goal) -> dict[str, object]:
        return {
            "id": goal.id,
            "name": goal.name,
            "context": goal.context,
            "tags": self._knowledge.tags_to_data(goal.tags),
        }

    def goal_from_data(self, data: dict[str, object]) -> Goal:
        return Goal(
            str(data["name"]),
            str(data["context"]),
            self._knowledge.tags_from_data(data.get("tags")),
            str(data.get("id", "")),
        )

    def preference_to_data(self, preference: Preference) -> dict[str, object]:
        return {
            "id": preference.id,
            "goal": preference.goal,
            "weight": preference.weight,
            "holder": list(preference.holder),
            "role": preference.role,
            "tags": self._knowledge.tags_to_data(preference.tags),
        }

    def preference_from_data(self, data: dict[str, object]) -> Preference:
        return Preference(
            str(data["goal"]),
            float(data["weight"]),  # type: ignore[arg-type]
            tuple(str(holder) for holder in data.get("holder", ())),  # type: ignore[union-attr]
            str(data.get("role", "")),
            self._knowledge.tags_from_data(data.get("tags")),
            str(data.get("id", "")),
        )
