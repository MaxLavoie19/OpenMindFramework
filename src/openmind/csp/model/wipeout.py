class Wipeout(Exception):
    """Propagation left a variable without any value."""

    def __init__(self, variable: str) -> None:
        super().__init__(f"{variable} has no value left")
        self.variable = variable
