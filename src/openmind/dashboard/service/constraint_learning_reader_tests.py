import json
from pathlib import Path

from openmind.dashboard.service.constraint_learning_reader import ConstraintLearningReader


def test_every_run_in_the_directory_is_read(tmp_path: Path) -> None:
    """Arms are launched together and each writes its own snapshot, so all of them are the news."""
    _wrote(tmp_path / "plain.json", position=435, at="2026-09-27 17:13:00")
    _wrote(tmp_path / "windowed.json", position=73, at="2026-09-27 17:13:44")
    found = ConstraintLearningReader().runs(tmp_path)
    assert tuple(one.run for one in found) == ("windowed", "plain"), "the run that spoke last comes first"
    assert tuple(one.position for one in found) == (73, 435)


def test_the_run_that_spoke_last_is_the_one_shown_when_none_is_asked_for(tmp_path: Path) -> None:
    """A run that finished two days ago must not be what the page shows while three others are running."""
    _wrote(tmp_path / "constraints.json", position=88, at="2026-09-25 18:09:37")
    _wrote(tmp_path / "windowed.json", position=73, at="2026-09-27 17:13:44")
    found = ConstraintLearningReader().latest(tmp_path)
    assert found is not None
    assert found.run == "windowed"


def test_a_run_can_be_asked_for_by_name(tmp_path: Path) -> None:
    _wrote(tmp_path / "plain.json", position=435, at="2026-09-27 17:13:00")
    _wrote(tmp_path / "windowed.json", position=73, at="2026-09-27 17:13:44")
    reader = ConstraintLearningReader()
    found = reader.latest(tmp_path, "plain")
    assert found is not None
    assert (found.run, found.position) == ("plain", 435)
    assert reader.latest(tmp_path, "nobody") is None, "a run nobody wrote is nothing, not the newest instead"


def test_a_file_that_is_not_a_snapshot_is_not_a_run(tmp_path: Path) -> None:
    """The directory a run logs to holds whatever else was put there, and none of it is a run."""
    (tmp_path / "settings.json").write_text(json.dumps({"workers": 8}))
    (tmp_path / "broken.json").write_text("{not json")
    _wrote(tmp_path / "plain.json", position=12, at="2026-09-27 17:00:00")
    assert tuple(one.run for one in ConstraintLearningReader().runs(tmp_path)) == ("plain",)


def test_nothing_said_reads_as_nothing(tmp_path: Path) -> None:
    assert ConstraintLearningReader().runs(tmp_path) == ()
    assert ConstraintLearningReader().latest(tmp_path) is None
    assert ConstraintLearningReader().runs(tmp_path / "never made") == ()


def test_a_run_that_does_not_say_when_is_timed_by_its_file(tmp_path: Path) -> None:
    """A run writing to a share writes a file whose time belongs to the machine holding it, so the run's own
    word is preferred — but a run that says nothing is still sortable."""
    where = tmp_path / "quiet.json"
    where.write_text(json.dumps({"position": 3}))
    found = ConstraintLearningReader().runs(tmp_path)
    assert len(found) == 1
    assert found[0].at, "a run that says nothing about when is timed by its file rather than left blank"


def _wrote(where: Path, position: int, at: str) -> None:
    where.write_text(
        json.dumps(
            {
                "position": position,
                "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                "rules": ["refused :- nothing"],
                "rightly refused": 14018,
                "let through": 329,
                "wrongly refused": 0,
                "rightly allowed": 53,
                "readings": 30,
                "seconds": 113.2,
                "at": at,
            }
        )
    )
