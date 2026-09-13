from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openmind.expression.model.expression import Expression


@dataclass(frozen=True, slots=True)
class StateVariable:
    """The value of a state variable; its indices complete the name, as in cell(row, col)."""

    base: str
    indices: tuple[Expression, ...] = ()
