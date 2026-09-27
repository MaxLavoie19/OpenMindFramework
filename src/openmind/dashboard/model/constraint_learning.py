from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConstraintLearning:
    """What a run learning a game has got to, as a page shows it.

    **Three things are learned from one walk and shown together.** What the game refuses, what a move does, and
    what its notation says are not separate exercises: a notation leaves out precisely what the rules make
    recoverable, so what it does not say is a statement about the rules, and what a move does is the thing the
    notation is about. `consequences` is what the predictor has concluded; `sorts`, `shapes` and `couplings` are
    the notation's own alphabet, the layouts it comes in, and what each place of each layout says.

    `position` is which position it is on and `fen` how that position is written; `picture` is it drawn, an SVG
    image, or empty for a game that does not draw itself. `rules` are the constraints as they now stand, most
    conditions first, each already readable.

    The four counts are the whole of how it is doing, and they are counted over every candidate the solver could
    propose — four thousand and ninety-six of them in chess, against twenty the game allows. `rightly_refused`
    and `rightly_allowed` are agreement; `let_through` is a candidate the game refuses and the constraints do not,
    which is a move OMF would offer and the game would reject; `wrongly_refused` is a move the game allows and the
    constraints refuse, which is a move OMF will never make and nothing will ever tell it about.

    `run` is which run said it, by the name of the file it writes. Several run at once and are meant to — an
    arm with a change against an arm without it is how anything here is decided — so a page showing one of them
    and calling it *the* run shows whichever was named first and hides the comparison the runs exist for."""

    position: int
    fen: str
    picture: str
    rules: tuple[str, ...]
    rightly_refused: int
    let_through: int
    wrongly_refused: int
    rightly_allowed: int
    seconds: float
    readings: int
    run: str = ""
    matching: int = 0
    written_by_hand: int = 0
    at: str = ""
    consequences: tuple[str, ...] = ()
    sorts: tuple[str, ...] = ()
    shapes: tuple[str, ...] = ()
    couplings: tuple[str, ...] = ()

    @property
    def matched(self) -> str:
        """How many of its constraints say what one of ours says, against how many were written by hand.

        Nothing compares the wording: two constraints refusing the same candidates are the same constraint, and
        OMF was never asked to arrive at ours. Empty where nobody wrote any down, which is the ordinary case for
        a game we have no rules for."""
        return f"{self.matching} / {self.written_by_hand}" if self.written_by_hand else ""

    @property
    def candidates(self) -> int:
        return self.rightly_refused + self.let_through + self.wrongly_refused + self.rightly_allowed

    @property
    def legal(self) -> int:
        """How many the game allows there."""
        return self.wrongly_refused + self.rightly_allowed
