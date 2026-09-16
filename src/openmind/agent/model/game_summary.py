from dataclasses import dataclass

from openmind.agent.model.model_description import ModelDescription
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl


@dataclass(frozen=True, slots=True)
class GameSummary:
    """A finished game, whatever played it, as it is remembered: the domain, the kind of game (see `agent_constant`), the
    round it belongs to (None outside training), its number within its kind and round, the seeds it was played from,
    the players' names with the model each player played in the same order, the final payoffs, the plies, why it ended
    and its record when the domain says, and on a clock, the time control, each step's seconds and budget (None where
    the player gives none), each player's clock at the end and the player whose time ran out."""

    domain: str
    kind: str
    round: int | None
    number: int
    seeds: tuple[int, ...]
    players: tuple[str, ...]
    models: tuple[ModelDescription, ...]
    payoffs: tuple[float, ...]
    plies: int
    ending: str | None = None
    record: str | None = None
    time_control: TimeControl | None = None
    seconds: tuple[float, ...] = ()
    budgets: tuple[float | None, ...] = ()
    clocks: tuple[Clock, ...] = ()
    flagged: str | None = None

    @property
    def label(self) -> str:
        """The game as its records name it: `round 1 arms game 12`, or `match game 3` outside training."""
        prefix = "" if self.round is None else f"round {self.round} "
        return f"{prefix}{self.kind} game {self.number}"
