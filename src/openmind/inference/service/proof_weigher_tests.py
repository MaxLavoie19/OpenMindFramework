import random

from openmind.inference.model.derivation import Derivation
from openmind.inference.model.derivation_step import GIVEN, RESOLVED, DerivationStep
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.proof_weigher import ProofWeigher
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal


def proof(*leaning: tuple[str, float]) -> Derivation:
    """A proof of one thing, leaning on those doubtful clauses."""
    steps = tuple(
        DerivationStep(number + 1, GIVEN, (), Clause((Literal("said", ()),), chance, name))
        for number, (name, chance) in enumerate(leaning)
    )
    reached = Clause((Literal("reached", ()),))
    steps = (*steps, DerivationStep(len(steps) + 1, RESOLVED, (), reached))
    return Derivation(reached, steps, tuple(name for name, _ in leaning))


def test_a_conclusion_nothing_reaches_has_no_chance_at_all() -> None:
    assert ProofWeigher().chance((), InferenceBudget(5.0)).value == 0.0


def test_a_proof_leaning_on_nothing_doubtful_is_certain() -> None:
    found = ProofWeigher().chance((proof(),), InferenceBudget(5.0))

    assert found.value == 1.0 and found.exact


def test_one_proof_is_worth_the_clause_it_leans_on() -> None:
    found = ProofWeigher().chance((proof(("a guess", 0.8)),), InferenceBudget(5.0))

    assert found.value == 0.8


def test_a_proof_leaning_on_two_clauses_needs_both_of_them() -> None:
    found = ProofWeigher().chance((proof(("one", 0.5), ("another", 0.5)),), InferenceBudget(5.0))

    assert abs(found.value - 0.25) < 1e-9


def test_two_proofs_leaning_on_different_clauses_are_two_independent_reasons() -> None:
    found = ProofWeigher().chance((proof(("one", 0.5)), proof(("another", 0.5))), InferenceBudget(5.0))

    assert abs(found.value - 0.75) < 1e-9


def test_two_proofs_leaning_on_the_same_clause_are_not_worth_more_than_one() -> None:
    found = ProofWeigher().chance((proof(("shared", 0.5)), proof(("shared", 0.5))), InferenceBudget(5.0))

    assert abs(found.value - 0.5) < 1e-9


def test_two_proofs_sharing_a_clause_are_worth_less_than_if_they_were_independent() -> None:
    weigher = ProofWeigher()
    shared = weigher.chance((proof(("shared", 0.5), ("mine", 0.5)), proof(("shared", 0.5), ("theirs", 0.5))), InferenceBudget(5.0))
    apart = weigher.chance((proof(("one", 0.5), ("mine", 0.5)), proof(("another", 0.5), ("theirs", 0.5))), InferenceBudget(5.0))

    assert abs(shared.value - 0.375) < 1e-9
    assert shared.value < apart.value


def test_a_proof_needing_less_makes_one_needing_more_of_the_same_add_nothing() -> None:
    weigher = ProofWeigher()
    alone = weigher.chance((proof(("one", 0.5)),), InferenceBudget(5.0))
    with_a_longer_way = weigher.chance((proof(("one", 0.5)), proof(("one", 0.5), ("another", 0.5))), InferenceBudget(5.0))

    assert abs(alone.value - with_a_longer_way.value) < 1e-9


def test_a_certain_proof_beside_a_doubtful_one_settles_it() -> None:
    found = ProofWeigher().chance((proof(), proof(("a guess", 0.5))), InferenceBudget(5.0))

    assert found.value == 1.0


def test_a_layout_too_large_for_the_budget_is_sampled_and_says_so() -> None:
    proofs = tuple(proof((f"clause {number}", 0.5), (f"clause {number + 1}", 0.5)) for number in range(12))

    found = ProofWeigher(draws=random.Random(7)).chance(proofs, InferenceBudget(5.0, nodes=4, draws=2000))

    assert not found.exact and found.spread > 0.0


def test_a_sampled_chance_lands_near_the_counted_one() -> None:
    proofs = (proof(("one", 0.5), ("mine", 0.5)), proof(("one", 0.5), ("theirs", 0.5)))

    counted = ProofWeigher().chance(proofs, InferenceBudget(5.0))
    sampled = ProofWeigher(draws=random.Random(11)).chance(proofs, InferenceBudget(5.0, nodes=1, draws=20_000))

    assert counted.exact and not sampled.exact
    assert abs(counted.value - sampled.value) < 0.05


def test_a_counted_chance_has_no_spread_because_it_was_not_estimated() -> None:
    found = ProofWeigher().chance((proof(("a guess", 0.8)),), InferenceBudget(5.0))

    assert found.spread == 0.0 and found.exact


def test_how_many_proofs_were_weighed_comes_back_with_the_chance() -> None:
    found = ProofWeigher().chance((proof(("one", 0.5)), proof(("another", 0.5))), InferenceBudget(5.0))

    assert found.derivations == 2


def test_proofs_resting_on_the_same_clauses_count_as_one_reason() -> None:
    weigher = ProofWeigher()

    assert weigher.independent((proof(("one", 0.5)), proof(("one", 0.5)))) == 1
    assert weigher.independent((proof(("one", 0.5)), proof(("another", 0.5)))) == 2


def test_how_many_draws_a_sampled_chance_takes_is_the_caller_s_to_set() -> None:
    proofs = (proof(("one", 0.5), ("mine", 0.5)), proof(("one", 0.5), ("theirs", 0.5)))

    few = ProofWeigher(draws=random.Random(3)).chance(proofs, InferenceBudget(5.0, nodes=1, draws=20))
    many = ProofWeigher(draws=random.Random(3)).chance(proofs, InferenceBudget(5.0, nodes=1, draws=20_000))

    assert many.spread < few.spread
