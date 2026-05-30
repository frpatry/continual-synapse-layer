"""Pytest configuration for the archived legacy test tree.

The tests in this directory and its subdirectories were written for
prior research directions (``continual_synapse`` and ``agi``) whose
source code now lives under ``src/legacy/``. The tests still reference
the OLD top-level import paths (``from continual_synapse...``,
``from agi...``), so importing them would fail at collection time.

Rather than rewrite the legacy tests to use the new ``legacy.``
namespace — which would change their on-disk content and break
``git log --follow`` semantics for readers — we tell pytest to skip
collection of any ``test_*.py`` in this directory entirely.

The files themselves remain on disk, fully readable as they were
written, with their original imports intact. They are preserved for
historical reference and project-evolution traceability, not for
execution.

To run a specific legacy test by hand (e.g. for a writeup), import
its package from the new location:

    PYTHONPATH=src python -c "from legacy.continual_synapse...; ..."

and execute the test function directly.
"""

from __future__ import annotations


# Skip collection of any test file directly in this directory.
# Subdirectories carry their own conftest.py with the same rule.
collect_ignore_glob = ["test_*.py"]
