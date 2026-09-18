import json
from pathlib import Path

from openmind.training.mapper.arm_library_json_mapper import ArmLibraryJsonMapper
from openmind.training.model.arm_library import ArmLibrary
from openmind.training.repository.arm_library_repository import ArmLibraryRepository

LIBRARY = ArmLibrary("chess", (("first", "chess/round 1"), ("second", "chess/round 2")))


def test_a_library_goes_to_json_and_comes_back() -> None:
    mapper = ArmLibraryJsonMapper()

    assert mapper.from_json(mapper.to_json(LIBRARY)) == LIBRARY


def test_a_library_written_before_arms_had_names_keeps_its_contexts() -> None:
    older = json.loads(ArmLibraryJsonMapper().to_json(LIBRARY))
    for item in older["arms"]:
        item["signal"] = item.pop("arm")

    assert ArmLibraryJsonMapper().from_json(json.dumps(older)) == LIBRARY


def test_the_repository_writes_a_library_and_loads_it_back(tmp_path: Path) -> None:
    repository = ArmLibraryRepository(ArmLibraryJsonMapper())

    path = repository.write(LIBRARY, tmp_path / "arms.json")

    assert repository.load(path) == LIBRARY
