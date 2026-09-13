from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmind.expression.model.expression import Expression


@dataclass(frozen=True, slots=True)
class Equals:
    """True when both sides have the same value."""

    left: Expression
    right: Expression
