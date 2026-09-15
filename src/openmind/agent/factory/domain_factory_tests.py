import re
from importlib.metadata import EntryPoint

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import VARIANTS as PRISONERS_DILEMMA_VARIANTS
from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.factory.prisoners_dilemma_factory import create_prisoners_dilemma_domain
from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain


def made_up_recipe(name: str) -> Domain:
    """An installed project's recipe, for the tests: a sudoku named as asked."""
    return create_sudoku_domain(name)


def install(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Makes the domain factory find made_up_recipe registered under each name."""
    registered = tuple(
        EntryPoint(name, "openmind.agent.factory.domain_factory_tests:made_up_recipe", "openmind.domains")
        for name in names
    )
    monkeypatch.setattr(
        "openmind.agent.factory.domain_factory.entry_points",
        lambda group: registered if group == "openmind.domains" else (),
    )


def test_creates_tictactoe() -> None:
    assert create_domain("tictactoe") == create_tictactoe_domain()


def test_creates_tictactoe_variants() -> None:
    assert create_domain("tictactoe/fourinarow") == create_tictactoe_domain(VARIANTS["fourinarow"])
    assert create_domain("tictactoe/gomoku") == create_tictactoe_domain(VARIANTS["gomoku"])
    assert create_domain("tictactoe/standard") == create_tictactoe_domain()


def test_creates_sudoku() -> None:
    assert create_domain("sudoku") == create_sudoku_domain()


def test_creates_the_prisoners_dilemma_and_its_variants() -> None:
    assert create_domain("prisonersdilemma") == create_prisoners_dilemma_domain()
    assert create_domain("prisonersdilemma/uncertain") == create_prisoners_dilemma_domain(PRISONERS_DILEMMA_VARIANTS["uncertain"])
    with pytest.raises(ValueError, match="Unknown variant 'endless' of prisonersdilemma; variants: standard, uncertain, simultaneous"):
        create_domain("prisonersdilemma/endless")


def test_unknown_variant_raises_with_the_variants() -> None:
    with pytest.raises(ValueError, match="Unknown variant 'chess960' of tictactoe; variants: standard, fourinarow, gomoku"):
        create_domain("tictactoe/chess960")


def test_an_installed_project_domain_is_created_by_its_recipe_with_the_whole_name(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "madeup")

    assert create_domain("madeup/fast") == create_sudoku_domain("madeup/fast")


def test_the_built_in_domains_come_before_installed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "tictactoe")

    assert create_domain("tictactoe") == create_tictactoe_domain()


def test_unknown_domain_raises_with_the_known_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "madeup")
    known = (
        "Unknown domain 'go'; known domains: tictactoe, tictactoe/fourinarow, tictactoe/gomoku, sudoku, prisonersdilemma, "
        "prisonersdilemma/uncertain, prisonersdilemma/simultaneous, rockpaperscissors, madeup"
    )

    with pytest.raises(ValueError, match=re.escape(known)):
        create_domain("go")
