"""Phase 1.1+1.2 AGI demo: multi-turn + cross-session + clean extraction.

Extends the Phase 1.0 demo with:

  - Multi-turn conversation: the assistant tracks prior turns
    inside a session via :class:`ConversationManager`.
  - Cross-session persistence: ``AGISystem.save`` writes the
    privacy-preserving memory state to disk; a fresh
    ``AGISystem.load`` after a simulated restart recovers the
    same memory.
  - Default LLM-driven fact extraction (Phase 1.2: now backed by
    Qwen2.5-1.5B-Instruct + few-shot prompt + heuristic gating
    for question/short inputs; regex remains as fallback).

Run from the repo root::

    python experiments/agi/phase_1_1_multi_turn.py

The script deletes ``/tmp/agi_demo_memory.json`` at the top so
re-runs always start from a clean state — earlier-phase demos
left garbage extractions in that file.

By default the foundation auto-picks CUDA + fp16 (Colab L4) or
CPU + fp32 (dev). On CPU the demo takes ~5-10 min on Qwen-1.5B —
most of the wall time goes to per-turn generation and per-turn
LLM extraction. On GPU it should be a few minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.foundation import FrozenFoundation  # noqa: E402
from agi.integration import AGISystem  # noqa: E402


SAVE_PATH = "/tmp/agi_demo_memory.json"


def _hr(title: str) -> None:
    print()
    print(f"--- {title} ---")


def _print_turn(result: dict) -> None:
    print(f"Assistant: {result['response']}")
    extracted = result.get("extracted_facts") or {}
    if extracted:
        print(f"  [Extracted this turn: {extracted}]")
    else:
        # Distinguish gated-by-heuristic vs nothing-to-extract.
        # The gated event was already printed inline by the
        # extractor; here we just note "no facts" for clarity.
        print("  [No facts extracted this turn]")
    if result.get("memory_used"):
        # Reduce noise: show merged facts (already deduped) and
        # the similarity of the top hit.
        merged = result.get("merged_facts") or {}
        top = result["retrieved_facts"][0] if result["retrieved_facts"] else None
        top_sim = top[1] if top is not None else None
        print(
            f"  [Used memory: {merged}"
            + (f"  top_sim={top_sim:.3f}" if top_sim is not None else "")
            + "]"
        )


def main() -> int:
    print("=== Phase 1.1+1.2 AGI Demo: Multi-Turn + Cross-Session ===\n")

    # Ensure a fresh start — previous runs may have left a memory
    # file behind with garbage data (e.g. Phase 1.1's confabulated
    # extractions). Phase 1.2 demos must start empty.
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print(f"[Removed stale save file: {SAVE_PATH}]\n")

    t_load = time.time()
    print("Loading foundation (Qwen2.5-1.5B-Instruct by default)... "
          "(first run downloads ~3 GB)")
    foundation = FrozenFoundation()
    print(
        f"Foundation loaded in {time.time() - t_load:.1f}s.  "
        f"model={foundation.model_name}  "
        f"device={foundation.device}  dtype={foundation.dtype}  "
        f"key_dim={foundation.key_dim}"
    )

    agi = AGISystem(foundation, use_llm_extraction=True)

    # ===== Session 1: introduction + chat =====
    _hr("Session 1: introduction + chat")
    agi.new_session()

    turns_session_1 = [
        "Bonjour, je m'appelle Francois.",
        "Je vis à Montréal et je travaille sur du continual learning.",
        "Tu peux me résumer ce qu'est l'apprentissage continu en quelques mots?",
        "Merci. Je préfère les explications courtes en général.",
    ]

    for turn in turns_session_1:
        print(f"\nUser: {turn}")
        t = time.time()
        result = agi.chat(turn, temperature=0.0)
        _print_turn(result)
        print(f"  [chat() took {time.time() - t:.1f}s]")

    agi.save(SAVE_PATH)
    print(f"\n  [Memory saved to {SAVE_PATH}]")
    print(f"  [Total entries: {len(agi.memory)}]")

    # ===== Simulated program restart =====
    _hr("Simulating program restart")
    del agi

    agi2 = AGISystem(foundation, use_llm_extraction=True)
    agi2.load(SAVE_PATH)
    print(
        f"  [Memory reloaded: {len(agi2.memory)} entries, "
        f"current_session={agi2.memory.current_session}]"
    )

    # ===== Session 2: continuation across restart =====
    _hr("Session 2 (after restart)")
    agi2.new_session()

    turns_session_2 = [
        "Bonjour, tu te souviens de moi?",
        "Quel est mon nom?",
        "Et où je vis?",
        "Sur quoi je travaille?",
    ]

    for turn in turns_session_2:
        print(f"\nUser: {turn}")
        t = time.time()
        result = agi2.chat(turn, temperature=0.0)
        _print_turn(result)
        print(f"  [chat() took {time.time() - t:.1f}s]")

    # ===== Privacy verification (post-persistence) =====
    _hr("Privacy verification (on-disk state)")
    with open(SAVE_PATH) as f:
        saved = json.load(f)
    print(f"Saved entries on disk: {len(saved['entries'])}")
    forbidden = ("raw_text", "original_input", "raw", "samples", "utterance")
    for i, entry in enumerate(saved["entries"]):
        print(
            f"  Entry {i}: keys={sorted(entry.keys())}  facts={entry['facts']}"
        )
        for k in forbidden:
            assert k not in entry, (
                f"Privacy violation on disk — entry {i} contains {k!r}"
            )
    print("  ✓ Persistence preserves privacy (no raw text on disk).")

    print("\n=== Demo complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
