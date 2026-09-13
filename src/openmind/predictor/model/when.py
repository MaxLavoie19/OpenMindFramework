from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmind.expression.model.expression import Expression
    from openmind.predictor.model.effect import Effect


@dataclass(frozen=True, slots=True)
class When:
    """Applies `then` when the condition is true, otherwise `otherwise`."""

    condition: Expression
    then: tuple[Effect, ...]
    otherwise: tuple[Effect, ...] = ()
