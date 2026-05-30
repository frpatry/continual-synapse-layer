"""Phase 1.0 AGI demo: fact retention across queries.

Wires Qwen-0.5B-Instruct (frozen) + structured-fact extractor +
privacy-preserving episodic memory and walks through four
scenarios that exercise the full observe → respond pipeline.

Scenarios:
1. Introduction — user states their name + location.
2. Retrieval — later query asks for the name; the augmented
   prompt must contain the name in the system context.
3. Control — fresh agent, same query, no prior introduction.
4. Multi-fact — user states name + age + location + preference
   in one utterance; follow-up summary query exercises the
   full merge path.

Final block: assert no raw-text fields ever landed in memory.

Run from the repo root::

    python experiments/agi/phase_1_0_fact_retention.py

By default the foundation auto-picks CUDA + fp16 (Colab L4) or
CPU + fp32 (dev). On CPU the demo takes ~1-2 min wall (mostly
generation); on GPU it's seconds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.foundation import FrozenFoundation  # noqa: E402
from agi.integration import AGISystem  # noqa: E402


def _hr(title: str) -> None:
    print()
    print(f"--- {title} ---")


def main() -> int:
    print("=== Phase 1.0 AGI Demo: Fact Retention ===\n")

    t_load = time.time()
    print("Loading Qwen-0.5B-Instruct foundation... "
          "(first run downloads ~1 GB)")
    foundation = FrozenFoundation()
    print(
        f"Foundation loaded in {time.time() - t_load:.1f}s.  "
        f"device={foundation.device}  dtype={foundation.dtype}  "
        f"key_dim={foundation.key_dim}"
    )

    agi = AGISystem(foundation)

    # ===== Scenario 1: introduction =====
    _hr("Scenario 1: User introduces themselves")
    intro = "Bonjour, mon nom est Francois et je vis à Montréal."
    print(f"User: {intro}")
    facts = agi.observe(intro)
    print(f"  [Extracted facts: {facts}]")
    print(f"  [Memory size: {len(agi.memory)}]")

    # ===== Scenario 2: retrieval test (the key demo) =====
    _hr("Scenario 2: Retrieval test (later query)")
    query = "Comment je m'appelle?"
    print(f"User: {query}")
    t_resp = time.time()
    result = agi.respond(query, temperature=0.0)  # greedy for reproducibility
    print(f"Assistant: {result['response']}")
    print(f"  [Memory used: {result['memory_used']}]")
    print(f"  [Retrieved facts: {result['retrieved_facts']}]")
    print(f"  [Merged facts: {result['merged_facts']}]")
    print(f"  [Generation: {time.time() - t_resp:.1f}s]")

    # ===== Scenario 3: control (no memory available) =====
    _hr("Scenario 3: Control — same query without prior introduction")
    fresh_agi = AGISystem(foundation)
    print(f"User: {query}")
    t_resp = time.time()
    result_control = fresh_agi.respond(query, temperature=0.0)
    print(f"Assistant: {result_control['response']}")
    print(f"  [Memory used: {result_control['memory_used']}]")
    print(f"  [Generation: {time.time() - t_resp:.1f}s]")

    # ===== Scenario 4: multi-fact retrieval =====
    _hr("Scenario 4: Multiple facts, mixed query")
    multi = (
        "Je m'appelle Marie, j'ai 30 ans et je vis à Paris. "
        "Je préfère les réponses courtes."
    )
    print(f"User: {multi}")
    facts = agi.observe(multi)
    print(f"  [Extracted: {facts}]")
    print(f"  [Memory size now: {len(agi.memory)}]")

    follow_up = "Peux-tu résumer ce que tu sais sur moi?"
    print(f"\nUser: {follow_up}")
    t_resp = time.time()
    result = agi.respond(follow_up, temperature=0.0)
    print(f"Assistant: {result['response']}")
    print(f"  [Retrieved: {result['retrieved_facts']}]")
    print(f"  [Merged: {result['merged_facts']}]")
    print(f"  [Generation: {time.time() - t_resp:.1f}s]")

    # ===== Privacy verification =====
    _hr("Privacy verification")
    print(f"Memory entries: {len(agi.memory)}")
    print("Stored fields per entry (asserting no raw text):")
    for i, entry in enumerate(agi.memory.entries):
        print(
            f"  Entry {i}: key.shape={tuple(entry.key.shape)}  "
            f"facts={entry.facts}  "
            f"created_session={entry.creation_session}  "
            f"access_count={entry.access_count}"
        )
        for forbidden in (
            "raw_text", "original_input", "raw", "samples", "utterance",
        ):
            assert not hasattr(entry, forbidden), (
                f"Privacy violation — entry {i} exposes {forbidden!r}"
            )
    print(
        "  ✓ No raw text fields present — only stable keys + "
        "structured facts."
    )

    print("\n=== Demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
