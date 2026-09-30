from openmind.dashboard.mapper.dashboard_html_mapper import DashboardHtmlMapper
from openmind.dashboard.model.constraint_learning import ConstraintLearning
from openmind.dashboard.model.game_view import GameView


def a_run(**how):
    held = {
        "position": 12,
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "picture": "",
        "rules": ("refused :- lands on(self, x, y, grid, outside)",),
        "rightly_refused": 13000,
        "let_through": 1300,
        "wrongly_refused": 0,
        "rightly_allowed": 20,
        "seconds": 16.2,
        "readings": 6,
    }
    return ConstraintLearning(**{**held, **how})


def test_the_page_shows_all_three_learnings_together():
    """What the game refuses, what a move does, and what its notation says are learned from one walk and shown
    side by side. They are not separate exercises: a notation leaves out what the rules make recoverable, so
    what it does not say is a statement about the rules."""
    learning = a_run(
        consequences=("moved grid at (the row of self, the column of self) onto (…)",),
        sorts=("x", "BNQR", "abcdefgh"),
        shapes=("[BNQR] [abcdefgh] [12345678]",),
        couplings=("'N' at place 1 of [BNQR] [abcdefgh] [12345678] says from type is 'knight'",),
    )

    page = DashboardHtmlMapper().constraints_page("chess", learning, 10)

    assert "What it refuses" in page and "lands on" in page
    assert "What a move does" in page and "moved grid at" in page
    assert "What the notation says" in page and "BNQR" in page
    assert "knight" in page


def test_what_has_not_been_learned_yet_says_so_rather_than_showing_nothing():
    """An empty panel reads as a broken page. That nothing has been worked out yet is news, and the reason it is
    empty is worth saying — a drawing needs two moves that differ, a symbol needs enough sightings to be more
    than an accident."""
    page = DashboardHtmlMapper().constraints_page("chess", a_run(), 10)

    assert "What a move does" in page and "Nothing yet" in page
    assert "What the notation says" in page


def test_a_run_that_has_said_nothing_at_all_is_not_an_error():
    """There is no page to show, which is itself the news."""
    page = DashboardHtmlMapper().constraints_page("chess", None, 10)

    assert "No run has said anything yet" in page


def test_every_page_carries_the_same_way_to_every_other():
    """A page reached only by whichever other page happened to link to it is a page found by knowing its
    address."""
    mapper = DashboardHtmlMapper()

    for page in (
        mapper.constraints_page("chess", a_run(), 10),
        mapper.constraints_page("chess", None, 10),
        mapper.missing_page("chess"),
        mapper.to_html(a_training(), 10),
    ):
        for where, name in mapper.PAGES:
            assert f"href='{where}'" in page and name in page


def a_training():
    """The training page's snapshot, which is the one page that builds its own shell — and so the one that was
    left without any way to reach the others."""
    from openmind.dashboard.model.dashboard_snapshot import DashboardSnapshot
    from openmind.dashboard.model.machine_status import MachineStatus

    machine = MachineStatus(
        memory_total=0, memory_available=0, swap_total=0, swap_free=0, processes=(), earlyoom=()
    )
    return DashboardSnapshot(
        domain="chess", taken_at="2026-09-24 18:00:00", progress=None, machine=machine
    )


def a_game(**held) -> GameView:
    """A game as a page shows it, with two sides that judged with different heuristics."""
    fields = {
        "label": "self-play game 7",
        "ended": "2026-09-25 22:00:00",
        "players": (("X", "position value"), ("O", "position value at price 0.01")),
        "payoffs": (1.0, 0.0),
        "ending": None,
        "record": None,
        "moves": ("place 2 2", "place 1 1"),
        "pictures": ("start", "after one", "after two"),
        "pictured": False,
        "id": "one",
        "heuristics": (
            ("X", "position value", (("sum(1 for at in here.piece if here.piece[at] == 'queen')", 9.5),)),
            ("O", "position value at price 0.01", (("here.mobility(me)", -0.25),)),
        ),
    }
    return GameView(**{**fields, **held})


def test_a_game_page_shows_the_rules_each_side_judged_with():
    """A name says which heuristic won; the rules say why, and reading them is the point of keeping them."""
    page = DashboardHtmlMapper().game_page("chess", a_game(), 30)

    assert "What each side judged with" in page
    assert "here.piece[at] == &#x27;queen&#x27;" in page
    assert "+9.5" in page and "-0.25" in page
    assert "Previous game" not in page  # it is the first, so there is nothing before it


def test_each_side_is_one_box_so_a_heading_stays_with_its_own_table():
    """A heading, a caption and a table laid beside each other are three things to a row of boxes and not one.

    Left loose, two sides came out as six items strung across the page, each heading beside somebody else's
    rules — nothing wrong with the numbers and the page unreadable. Pinned by counting the boxes, because that
    is the thing that was missing rather than anything the boxes contain."""
    page = DashboardHtmlMapper().game_page("chess", a_game(), 30)

    assert page.count("<div class='judged'>") == 2
    assert "<div class='heuristics'><div class='judged'><h3>" in page


def test_a_side_that_judged_with_nothing_says_so_rather_than_showing_an_empty_table():
    page = DashboardHtmlMapper().game_page(
        "chess", a_game(heuristics=(("X", "self-play", ()),)), 30
    )

    assert "It judged with nothing" in page or "judged with nothing" in page


def test_a_game_page_links_to_the_games_either_side_of_it():
    page = DashboardHtmlMapper().game_page("chess", a_game(previous_id="before", next_id="after"), 30)

    assert "/game/before'>Previous game" in page
    assert "/game/after'>Next game" in page
