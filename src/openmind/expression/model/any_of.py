from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmind.expression.model.expression import Expression


@dataclass(frozen=True, slots=True)
class AnyOf:
    """True when at least one operand is true."""

    operands: tuple[Expression, ...]
