import os
from pathlib import Path

from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader
from openmind.dashboard.service.log_progress_reader import LogProgressReader

ROUND_1 = [
    "INFO  __main__ Training in 14 worker processes",
    "INFO  openmind.training.service.value_training_loop Round 1 of 100000: self-play valuing positions with the start rules",
    "INFO  openmind.mcts.service.tree_search Searching 100 iterations for white",
    "INFO  openmind.inference.service.position_deducer Deduced move(uci='e1f1') for white within 2 plies: payoffs white=0.0 black=1.0",
    "INFO  openmind.mcts.service.tree_search Searching 100 iterations for black",
    "INFO  openmind.training.service.self_play Self-play game with seeds 11 and 12 finished in 120 plies: 120 samples, payoffs white=0.5 black=0.5",
    "INFO  openmind.training.service.self_play Self-play game with seeds 13 and 14 finished in 98 plies: 98 samples, payoffs white=1.0 black=0.0",
    "INFO  openmind.evaluation.service.match_runner Game with seeds 15 and 16 finished in 40 plies, the evaluated policy playing white: payoffs white=1.0 black=0.0",
]


def write(path: Path, lines: list[str], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as file:
        file.write("".join(f"{line}\n" for line in lines))


def test_progress_counts_the_round_s_games_searches_and_deductions_and_keeps_the_notable_lines(tmp_path: Path) -> None:
    write(tmp_path / "run.log", ROUND_1)

    progress = LogProgressReader(IncrementalLineReader()).progress(tmp_path)

    assert progress is not None
    assert (progress.round, progress.rounds, progress.games, progress.matches, progress.searched, progress.deduced) == (
        1,
        100000,
        2,
        1,
        2,
        1,
    )
    assert (progress.draws, progress.decisive) == (1, 1)
    assert progress.round_note == "self-play valuing positions with the start rules"
    assert progress.recent == (
        "INFO __main__: Training in 14 worker processes",
        "INFO value_training_loop: Round 1 of 100000: self-play valuing positions with the start rules",
    )


def test_progress_picks_up_where_it_stopped_and_starts_counting_again_at_a_new_round(tmp_path: Path) -> None:
    log, reader = tmp_path / "run.log", LogProgressReader(IncrementalLineReader())
    write(log, ROUND_1)
    reader.progress(tmp_path)

    write(log, ["INFO  openmind.training.service.self_play Self-play game with seeds 17 and 18 finished in 80 plies: 80 samples, payoffs white=0.5 black=0.5"], "a")
    third = reader.progress(tmp_path)
    write(log, ["INFO  openmind.training.service.value_training_loop Round 2 of 100000: self-play valuing positions with round 1's rules"], "a")
    second_round = reader.progress(tmp_path)

    assert third is not None and (third.games, third.draws, third.decisive) == (3, 2, 1)
    assert second_round is not None
    assert (
        second_round.round,
        second_round.games,
        second_round.matches,
        second_round.searched,
        second_round.deduced,
        second_round.draws,
        second_round.decisive,
    ) == (2, 0, 0, 0, 0, 0, 0)


def test_a_game_between_arms_counts_by_its_payoffs_too(tmp_path: Path) -> None:
    write(
        tmp_path / "run.log",
        [
            "INFO  openmind.training.service.self_play Self-play game with seeds 1 and 2, uniform against win, finished in 80 plies by checkmate: 0 samples, payoffs white=0.0 black=1.0",
            "INFO  openmind.training.service.self_play Self-play game with seeds 1 and 2 record: [Event \"?\"] 1. e4 e5 *",
        ],
    )

    progress = LogProgressReader(IncrementalLineReader()).progress(tmp_path)

    assert progress is not None and (progress.games, progress.draws, progress.decisive) == (1, 0, 1)


def test_a_newer_log_starts_over_and_a_directory_without_logs_gives_nothing(tmp_path: Path) -> None:
    reader = LogProgressReader(IncrementalLineReader())
    older, newer = tmp_path / "older.log", tmp_path / "newer.log"
    write(older, ROUND_1)
    reader.progress(tmp_path)

    write(newer, ["INFO  __main__ Training in 14 worker processes"])
    os.utime(older, (1, 1))
    progress = reader.progress(tmp_path)

    assert progress is not None and (progress.path, progress.round, progress.games) == (newer, None, 0)
    assert LogProgressReader(IncrementalLineReader()).progress(tmp_path / "empty") is None


ROUND_2 = [
    "2026-09-16 01:18:31,678 INFO  openmind.training.service.value_training_loop Round 2 of 100000: self-play valuing positions with round 1's rules",
    "2026-09-16 01:19:02,101 INFO  openmind.training.service.self_play Self-play game with seeds 21 and 22 finished in 40 plies by checkmate: 40 samples, payoffs white=1.0 black=0.0",
    "2026-09-16 01:20:11,900 INFO  openmind.training.service.self_play Self-play game with seeds 23 and 24 finished in 60 plies by threefold repetition: 60 samples, payoffs white=0.5 black=0.5",
    "2026-09-16 01:21:44,002 INFO  openmind.training.service.self_play Self-play game with seeds 25 and 26 finished in 80 plies by checkmate: 80 samples, payoffs white=0.0 black=1.0",
]


def test_a_line_is_read_whether_or_not_it_carries_the_time_it_was_produced_at(tmp_path: Path) -> None:
    write(tmp_path / "run.log", [*ROUND_1, *ROUND_2])

    progress = LogProgressReader(IncrementalLineReader()).progress(tmp_path)

    assert progress is not None
    assert (progress.round, progress.games, progress.decisive, progress.draws) == (2, 3, 2, 1)
    assert progress.round_note == "self-play valuing positions with round 1's rules"


def test_every_round_keeps_what_its_games_came_to_while_the_round_being_played_grows(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    write(log, ROUND_1)
    reader = LogProgressReader(IncrementalLineReader())
    reader.progress(tmp_path)

    (first,) = reader.played()

    assert (first.number, first.games, first.decisive, first.draws) == (1, 2, 1, 1)
    assert (first.plies, first.shortest, first.longest, first.mean_plies) == (218, 98, 120, 109.0)
    assert first.endings == ()  # This domain's games don't say why they ended.

    write(log, ROUND_2, mode="a")
    os.utime(log, (0, 0))
    reader.progress(tmp_path)

    first, second = reader.played()

    assert (first.games, second.games) == (2, 3)  # The first round keeps what it came to.
    assert (second.decisive, second.draws, second.mean_plies) == (2, 1, 60.0)
    assert second.endings == (("checkmate", 2), ("threefold repetition", 1))
    assert (second.shortest, second.longest, round(second.decisive_share, 2)) == (40, 80, 0.67)


def test_a_newer_log_starts_the_rounds_over(tmp_path: Path) -> None:
    write(tmp_path / "first.log", ROUND_1)
    reader = LogProgressReader(IncrementalLineReader())
    reader.progress(tmp_path)

    write(tmp_path / "second.log", ROUND_2)
    reader.progress(tmp_path)

    (only,) = reader.played()
    assert (only.number, only.games) == (2, 3)


def test_the_notable_lines_keep_the_date_and_time_they_were_produced_at(tmp_path: Path) -> None:
    write(
        tmp_path / "run.log",
        [
            "2026-09-16 19:03:08,700 INFO  openmind.training.service.value_training_loop Round 1 of 100000: self-play without value rules",
            "2026-09-16 19:03:09,012 INFO  openmind.parallel.service.memory_guard Cut the caches back to 567692 entries of 667879",
        ],
    )

    progress = LogProgressReader(IncrementalLineReader()).progress(tmp_path)

    assert progress is not None
    assert progress.round == 1
    assert progress.recent[0] == "2026-09-16 19:03:08 INFO value_training_loop: Round 1 of 100000: self-play without value rules"
