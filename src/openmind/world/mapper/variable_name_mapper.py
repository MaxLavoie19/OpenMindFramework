from openmind.world.model.value import Value


class VariableNameMapper:
    """Maps a base name and its indices to a variable name, ("cell", (2, 3)) -> "cell(2,3)", and back."""

    def to_name(self, base: str, indices: tuple[Value, ...]) -> str:
        if not indices:
            return base
        return f"{base}({','.join(str(index) for index in indices)})"

    def from_name(self, name: str) -> tuple[str, tuple[str, ...]]:
        """The base and the index texts of a variable name: "cell(2,3)" -> ("cell", ("2", "3"))."""
        if not name.endswith(")") or "(" not in name:
            return name, ()
        base, _, indices = name[:-1].partition("(")
        return base, tuple(indices.split(","))
