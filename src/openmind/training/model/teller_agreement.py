from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TellerAgreement:
    """One heuristic put beside a teller: how closely what it makes of a position tracks what the teller does.

    **The dense measure beside the sparse one.** `Agreement` waits for a game to finish and then credits one
    number back across eighty plies; this gives a number per position, so a heuristic can be told hopeless on
    a handful of boards rather than on a handful of games. That is the whole of what it buys — evidence per
    observation, where the payoff gives evidence per game.

    **And it cannot be the thing that decides.** A heuristic that tracks a teller perfectly has learned the
    teller, blind spots and all, and what is wanted is winning. So this narrows the field and the payoff picks
    among what it leaves: outside knowledge enters as a claim with measured reliability, never as truth.

    **Four facts and no score, for the reason `Agreement` gives.** A share can be read off these several ways
    and they mean different things, and folding them into one number here would make that choice on behalf of
    whoever is selecting.

    `agreed` is how closely the two move together over the positions this heuristic had an opinion about,
    from -1 to 1. Nought is the line it is read against and needs no baseline carried beside it: a heuristic
    with no opinion tracks nothing, so nought is what knowing nothing looks like. Below it, the heuristic is
    not merely uninformed but pointed the wrong way.

    `decided` is how many positions it answered, `declined` where it knew nothing, and `undecided` where its
    opinion separated nothing — the same three, meaning the same things, as on `Agreement`. Kept apart from
    `agreed` so that coverage is read beside the measure and never folded into it.

    `told` is what the teller was asked about, so that a heuristic answering ten of a thousand is not read as
    one answering ten of ten."""

    holder: str
    agreed: float
    decided: int
    declined: int
    undecided: int
    told: int
