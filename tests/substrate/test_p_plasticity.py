"""Tests for P-P Hebbian plasticity (creation gate + age-modulated decay)."""

from __future__ import annotations

from substrate.p_connectivity import PConnectivity
from substrate.p_entity import PEntity
from substrate.p_plasticity import apply_pp_plasticity


def _three_p(act0: float, act1: float, act2: float) -> dict[int, PEntity]:
    """Helper: 3 P entities with given activations.

    The covariance Hebbian rule uses ``mean²`` as a baseline, so a
    2-P pool with equal activations always yields δ = 0. Tests use 3+
    P with at least one quiet one to pull the mean down, mirroring
    the live substrate's typical state."""
    return {
        0: PEntity(id=0, components=(0, 1), activation=act0),
        1: PEntity(id=1, components=(2, 3), activation=act1),
        2: PEntity(id=2, components=(4, 5), activation=act2),
    }


# ---------- no-op cases ----------


def test_pp_plasticity_no_change_if_zero_activation():
    """All P quiet → no connection touched / created."""
    p_entities = _three_p(0.0, 0.0, 0.0)
    pc = PConnectivity()
    apply_pp_plasticity(
        p_entities, pc, system_age=10.0,
        eta_pp=0.1, lambda_pp_decay=0.01, min_coactivation_to_create=0.1,
    )
    assert pc.connection_count() == 0


def test_pp_plasticity_no_change_with_single_p():
    """With < 2 P, the routine early-exits — never touches the store."""
    p_entities = {0: PEntity(id=0, components=(0, 1), activation=1.0)}
    pc = PConnectivity()
    pc.update_weight(0, 99, 0.5)  # spurious entry — left untouched
    apply_pp_plasticity(p_entities, pc, system_age=10.0, eta_pp=0.1)
    assert pc.get_weight(0, 99) == 0.5


# ---------- creation gate ----------


def test_pp_plasticity_creates_connection_above_coact_threshold():
    """Two strongly co-active P + a quiet one → new connection forms."""
    p_entities = _three_p(0.8, 0.8, 0.0)
    pc = PConnectivity()
    apply_pp_plasticity(
        p_entities, pc, system_age=10.0,
        eta_pp=0.1, min_coactivation_to_create=0.1,
    )
    # 0.8 · 0.8 = 0.64 > min_coact (0.1)
    # mean = (0.8 + 0.8 + 0.0) / 3 ≈ 0.533 → mean² ≈ 0.284
    # δ = 0.1 · (0.64 − 0.284) ≈ 0.036 > 0 → create.
    assert pc.get_weight(0, 1) > 0.0
    # Pairs involving the quiet entity have a · b = 0 < min_coact.
    assert pc.get_weight(0, 2) == 0.0
    assert pc.get_weight(1, 2) == 0.0


def test_pp_plasticity_does_not_create_below_threshold():
    """Weakly co-active P → no new connection (gate prevents noise edges)."""
    p_entities = _three_p(0.2, 0.2, 0.0)
    pc = PConnectivity()
    apply_pp_plasticity(
        p_entities, pc, system_age=10.0,
        eta_pp=0.1, min_coactivation_to_create=0.1,
    )
    # 0.2 · 0.2 = 0.04 < 0.1 → gate blocks creation.
    assert pc.get_weight(0, 1) == 0.0
    assert pc.connection_count() == 0


# ---------- strengthening / decay of existing connections ----------


def test_pp_plasticity_strengthens_existing_connection():
    """Existing connection + co-activation → weight grows."""
    p_entities = _three_p(0.8, 0.8, 0.0)
    pc = PConnectivity()
    pc.update_weight(0, 1, 0.1)
    initial = pc.get_weight(0, 1)
    apply_pp_plasticity(
        p_entities, pc, system_age=1.0,
        eta_pp=0.1, lambda_pp_decay=0.0001,  # tiny decay so growth dominates
    )
    assert pc.get_weight(0, 1) > initial


def test_pp_plasticity_decay_applies_to_existing():
    """Existing connection + no co-activation → weight shrinks."""
    p_entities = _three_p(0.0, 0.0, 0.0)
    pc = PConnectivity()
    pc.update_weight(0, 1, 0.5)
    initial = pc.get_weight(0, 1)
    apply_pp_plasticity(
        p_entities, pc, system_age=1.0,
        eta_pp=0.1, lambda_pp_decay=0.1,
    )
    assert pc.get_weight(0, 1) < initial


def test_pp_plasticity_age_modulates_decay():
    """Same decay setup at age=1 vs age=1000 → older substrate decays less."""
    def run_at_age(age: float) -> float:
        p_entities = _three_p(0.0, 0.0, 0.0)
        pc = PConnectivity()
        pc.update_weight(0, 1, 1.0)
        apply_pp_plasticity(
            p_entities, pc, system_age=age,
            eta_pp=0.0, lambda_pp_decay=0.1,
        )
        return pc.get_weight(0, 1)

    w_young = run_at_age(1.0)
    w_old = run_at_age(1000.0)
    assert w_young < 1.0, "decay should reduce a 1.0 weight at age=1"
    assert w_old < 1.0, "decay should reduce a 1.0 weight at age=1000"
    assert w_old > w_young, (
        f"older substrate decays less than younger: "
        f"young={w_young:.4f}, old={w_old:.4f}"
    )


def test_pp_plasticity_dropped_connection_when_decayed_to_zero():
    """Weight clipped to zero via decay → entry removed from store."""
    p_entities = _three_p(0.0, 0.0, 0.0)
    pc = PConnectivity()
    pc.update_weight(0, 1, 0.01)  # very small
    apply_pp_plasticity(
        p_entities, pc, system_age=1.0,
        eta_pp=0.0, lambda_pp_decay=10.0,  # massive decay
    )
    # Decay overwhelms the weight → clipped to 0 → entry dropped.
    assert pc.get_weight(0, 1) == 0.0
    assert pc.connection_count() == 0
