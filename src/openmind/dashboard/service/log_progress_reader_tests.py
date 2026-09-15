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

    assert third is not None and third.games == 3
    assert second_round is not None
    assert (second_round.round, second_round.games, second_round.matches, second_round.searched, second_round.deduced) == (2, 0, 0, 0, 0)


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
