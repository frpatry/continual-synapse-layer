"""Skip collection of archived AGI-era tests.

See ``tests/legacy/conftest.py`` for the rationale: legacy tests
keep their original imports (``from agi...``) and would fail at
collection time if pytest tried to load them.
"""

from __future__ import annotations


collect_ignore_glob = ["test_*.py"]
