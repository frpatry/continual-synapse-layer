"""PEntity — emerged binary relation between two N.

Per THEORY.md §2.1 and the coexistence resolution (Q1 = B): when a
pair of N entities reach the emergence criteria, a first-class P
entity is created. The underlying weight ``W[i, j]`` in the
connectivity matrix continues to exist — P does not replace it.
P lives at a higher abstraction level on top of the same fabric.

Phase 2a scope (observational): P is created, decayed, and dissolved,
but does NOT participate in N-level dynamics. Its activation is
purely derived from the activations of its component N. Phase 2b
will introduce P-P connections and let P influence N propagation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PEntity:
    """Emerged pair entity (P-level).

    Attributes:
        id: stable integer identifier assigned by the substrate.
        components: ``(n_i, n_j)`` ids of the two N this P binds.
            Stored in canonical sorted order (``components[0] <
            components[1]``) so a pair has exactly one representation
            regardless of which order it was emerged with.
        activation: current activation in ``[0, 1]``. In Phase 2a,
            derived each step as the mean activation of its components.
        weight: structural weight in ``[0, ∞)``. Initialised to 1.0
            on emergence (a P is "mature" at birth — it had to clear
            the emergence threshold to exist at all). Grows with use,
            decays with age, dissolves when it falls below the
            substrate's viability threshold.
        age_at_emergence: ``system_age`` of the substrate at the
            moment of emergence — kept for diagnostics + later analyses.
    """

    id: int
    components: tuple[int, int]
    activation: float = 0.0
    weight: float = 1.0
    age_at_emergence: float = 0.0
    protected_until: int = 0
    """Step count below which this P entity is exempt from dissolution.

    Phase 6f mechanism: newly-emerged P entities are given a protection
    window (``K_protect`` steps from :attr:`Substrate.k_protect`) during
    which their weight can drop below ``p_viability_threshold`` without
    causing dissolution. This gives a freshly-formed attractor time to
    consolidate via subsequent training cycles + consolidation phases
    before being subjected to dissolution competition from older,
    more-established attractors. Bio-inspired analog: protein-synthesis-
    dependent LTP late phase + synaptic tagging — newly-potentiated
    synapses are actively maintained for ~30 min – 1 h post-induction
    in mammalian neurons."""

    def __post_init__(self) -> None:
        # Canonicalise: components are an unordered pair, so we sort
        # them on construction so equality / set membership / dict
        # keys all work without the caller having to remember.
        a, b = self.components
        if a > b:
            self.components = (b, a)

    def reset_activation(self) -> None:
        """Zero out the current activation (mirrors :meth:`N.reset_activation`)."""
        self.activation = 0.0

    def is_protected(self, current_step: int) -> bool:
        """True if this P is currently within its post-emergence
        protection window.

        Substrate.``_decay_and_dissolve_p`` honours this — protected
        P entities still see decay applied to their weight, but the
        check that triggers dissolution at ``weight < viability`` is
        skipped. Once ``current_step >= protected_until``, normal
        dissolution dynamics resume.
        """
        return current_step < self.protected_until
