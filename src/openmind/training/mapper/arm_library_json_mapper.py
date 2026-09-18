import json

from openmind.training.model.arm_library import ArmLibrary


class ArmLibraryJsonMapper:
    """Maps an arm library to JSON text and back: every arm with the context whose rules it plays. Reading a library
    written before arms had contexts, under the older `signal` key, keeps working."""

    def to_json(self, library: ArmLibrary) -> str:
        return json.dumps(
            {
                "domain": library.domain,
                "arms": [{"arm": name, "context": context} for name, context in library.contexts],
            },
            indent=2,
        )

    def from_json(self, text: str) -> ArmLibrary:
        data = json.loads(text)
        contexts = tuple(
            (item.get("arm", item.get("signal")), item["context"]) for item in data.get("arms", [])
        )
        return ArmLibrary(data["domain"], contexts)
