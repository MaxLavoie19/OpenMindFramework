import re

import pytest

from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain


def test_creates_tictactoe() -> None:
    assert create_domain("tictactoe") == create_tictactoe_domain()


def test_creates_tictactoe_variants() -> None:
    assert create_domain("tictactoe/fourinarow") == create_tictactoe_domain(VARIANTS["fourinarow"])
    assert create_domain("tictactoe/gomoku") == create_tictactoe_domain(VARIANTS["gomoku"])
    assert create_domain("tictactoe/standard") == create_tictactoe_domain()


def test_creates_sudoku() -> None:
    assert create_domain("sudoku") == create_sudoku_domain()


def test_unknown_variant_raises_with_the_variants() -> None:
    with pytest.raises(ValueError, match="Unknown variant 'chess960' of tictactoe; variants: standard, fourinarow, gomoku"):
        create_domain("tictactoe/chess960")


def test_unknown_domain_raises_with_the_known_domains() -> None:
    known = "Unknown domain 'chess'; known domains: tictactoe, tictactoe/fourinarow, tictactoe/gomoku, sudoku"

    with pytest.raises(ValueError, match=re.escape(known)):
        create_domain("chess")
