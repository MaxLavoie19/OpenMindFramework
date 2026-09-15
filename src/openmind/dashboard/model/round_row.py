from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoundRow:
    """A finished round as the page shows it: its number, its value rules, the chosen fit's held-out loss, the held-out
    error, its games against each opponent as (opponent, "wins / draws / losses"), against the previous round's agent
    (None without), its pondering as (positions pondered, proven, seeds, seeds kept by the search, seeds in the rules)
    (None without), and its seconds."""

    number: int
    rules: int
    held_out_loss: float | None
    held_out_error: float | None
    baselines: tuple[tuple[str, str], ...]
    against_previous: str | None
    pondering: tuple[int, int, int, int, int] | None
    seconds: float
