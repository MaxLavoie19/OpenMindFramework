from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Agreement:
    """One heuristic put to one game's decisions: how much of what happened it expected, and how often it had
    anything to say at all.

    **Four facts and no score, because what they are worth is the caller's.** A share can be read off these
    several ways and they mean different things — the mass over the decisions it answered is how well it
    predicts when it speaks, and the same mass over every decision is how well it would play a whole game.
    Folded into one number here, the choice between them would be made by whoever wrote this rather than by
    whoever is selecting.

    `mass` is what it expected of what actually happened: for each decision it answered, the probability it
    gave the action taken, weighted by what the game paid the player who took it. A heuristic that sided with
    the winner gathers that mass at full weight and one that sided with the loser gathers none of it, which is
    what makes this about winning rather than only about predicting.

    `decided` is how many decisions it answered — where it rated the actions on offer and did not rate them
    all alike. `declined` is where it knew nothing, and `undecided` where it had an opinion that separated
    nothing. Neither is a disagreement: not firing is not the same as being wrong, and a rule that fires
    rarely and is right is a rule worth keeping. Kept apart from `decided` so that coverage is read beside the
    score and never folded into it — a detector held to a policy's coverage is a detector marked down for
    being a detector.

    `offered` is the mass the same decisions would have given a heuristic with no opinion at all: one over how
    many actions were on offer, weighted the same way. It is the line the score is read against. Above it, the
    heuristic saw something; at it, it knows nothing; below it, it is actively wrong about what wins.
    """

    holder: str
    mass: float
    decided: int
    declined: int
    undecided: int
    offered: float
