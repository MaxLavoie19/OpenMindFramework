from dataclasses import dataclass

from openmind.training.model.self_play_settings import SelfPlaySettings


@dataclass(frozen=True, slots=True)
class RankingSettings:
    """How heuristics are ranked against each other: how many games are played in all, what each one may spend, and
    how much of the choosing is exploring.

    `exploration` is UCB1's weight on how little a heuristic has been tried against how well it has done. Higher tries
    the unpromising for longer, which is what a bootstrap wants: a heuristic that lost its first game may have been
    unlucky."""

    play: SelfPlaySettings
    games: int = 20
    exploration: float = 1.4142135623730951
    seed: int = 0
