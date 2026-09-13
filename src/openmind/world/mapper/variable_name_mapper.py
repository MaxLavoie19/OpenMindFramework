from openmind.world.model.value import Value


class VariableNameMapper:
    """Maps a base name and its indices to a variable name: ("cell", (2, 3)) -> "cell(2,3)"."""

    def to_name(self, base: str, indices: tuple[Value, ...]) -> str:
        if not indices:
            return base
        return f"{base}({','.join(str(index) for index in indices)})"
