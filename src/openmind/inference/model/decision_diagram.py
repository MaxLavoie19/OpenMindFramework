from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecisionNode:
    """One doubtful clause, and what follows in each case: that it held, and that it did not.

    This is the shape that gets shared premises right, and getting them right is the whole reason it exists.
    A conclusion reached two ways is not twice as well supported when both ways lean on the same doubtful clause —
    it is the same support counted twice. Multiplying the two proofs' chances says otherwise, and says it
    confidently.

    Splitting on the clause instead makes the double-counting impossible to write. Ask what follows if it held and
    what follows if it did not; in each branch it has one answer, so no branch can use it twice. Branches that come
    out the same are one node, which is what keeps this from growing as fast as the proofs do."""

    variable: str
    when_held: "DecisionDiagram"
    when_not: "DecisionDiagram"


#: What a set of proofs comes to, once it is laid out so the clauses they share are shared.
#:
#: `True` where the conclusion follows whatever else happens, `False` where nothing left can reach it, and a node
#: where it turns on one doubtful clause.
type DecisionDiagram = bool | DecisionNode
