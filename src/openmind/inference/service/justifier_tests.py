from openmind.epistemology.service.foundherentism import Foundherentism
from openmind.inference.model.derivation import Derivation
from openmind.inference.model.derivation_step import GIVEN, DerivationStep
from openmind.inference.service.justifier import Justifier
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal


def resting_on(*names: str) -> Derivation:
    steps = tuple(
        DerivationStep(number + 1, GIVEN, (), Clause((Literal("said", ()),), 1.0, name))
        for number, name in enumerate(names)
    )
    return Derivation(Clause(), steps)


def justifier() -> Justifier:
    return Justifier(Foundherentism())


def test_two_proofs_from_different_clauses_are_two_reasons() -> None:
    found = justifier().independent((resting_on("one way"), resting_on("another way")))

    assert found == 2


def test_two_proofs_from_the_same_clauses_are_one_reason_found_twice() -> None:
    found = justifier().independent((resting_on("one way"), resting_on("one way")))

    assert found == 1


def test_the_order_clauses_were_used_in_does_not_make_a_second_reason() -> None:
    found = justifier().independent((resting_on("first", "second"), resting_on("second", "first")))

    assert found == 1


def test_a_proof_resting_on_nothing_named_is_no_reason_of_its_own() -> None:
    assert justifier().independent((Derivation(Clause()),)) == 0


def test_two_proofs_sharing_one_clause_but_not_the_other_are_still_two_reasons() -> None:
    found = justifier().independent((resting_on("shared", "mine"), resting_on("shared", "theirs")))

    assert found == 2
