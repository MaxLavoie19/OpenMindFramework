from openmind.structure.model.scalar import Scalar


def test_a_scalar_holds_one_value() -> None:
    assert Scalar("X").with_value("O") == Scalar("O")
