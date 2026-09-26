from dataclasses import dataclass

from openmind.training.model.self_play_settings import SelfPlaySettings


@dataclass(frozen=True, slots=True)
class RankingSettings:
    """How heuristics are ranked against each other: how many games are played in all, what each one may spend, and
    how much of the choosing is exploring.

    `exploration` is UCB1's weight on how little a heuristic has been tried against how well it has done. Higher tries
    the unpromising for longer, which is what a bootstrap wants: a heuristic that lost its first game may have been
    unlucky.

    `games` is None to play until something stops it, which is what ranking wants when nobody knows how many games
    it takes: a bandit has no natural stopping point, and a number picked here would be a budget invented by the
    thing spending it. `standings` is how often the table so far is written out along the way, since a run with no
    end has to be readable while it runs."""

    play: SelfPlaySettings
    games: int | None = 20
    exploration: float = 1.4142135623730951
    seed: int = 0
    standings: int = 10
