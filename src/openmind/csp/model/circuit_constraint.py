from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CircuitConstraint:
    """Parameters whose values form one cycle through every one of them.

    Position `i` is `variables[i]`, and the value `i` means that position — so a value names a variable by where it
    stands in this tuple, which is the only way a constraint can name one: a rule is handed values and never learns
    which parameter each came from. The order here is the operand order of the `circuit(...)` call that declared
    it, never the order its parameters happen to be declared in.

    It says the successors are all different *and* that they make a single cycle rather than several, so an
    all-different group beside it says nothing it does not already say."""

    variables: tuple[str, ...]
