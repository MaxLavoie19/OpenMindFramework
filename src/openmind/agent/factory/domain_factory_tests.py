import pytest

from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain


def test_creates_tictactoe() -> None:
    assert create_domain("tictactoe") == create_tictactoe_domain()


def test_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="chess"):
        create_domain("chess")
