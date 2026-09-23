"""Shared fixtures.

Reference meshes are fetched on demand, not committed, so tests that need a
compiled model skip with an actionable message rather than failing when a fresh
clone has not run the fetch script yet.
"""

import pytest

from robotic_arm.reference import BASELINE_MJCF, REFERENCE_DIR, ReferenceMissingError


@pytest.fixture(scope="session")
def baseline():
    """The stock Menagerie model, or a skip if reference files are absent."""
    from robotic_arm.reference import load_baseline

    if not (REFERENCE_DIR / "mjcf" / "assets").exists():
        pytest.skip("run: uv run python scripts/fetch_reference.py")
    try:
        return load_baseline()
    except ReferenceMissingError as exc:
        pytest.skip(str(exc))
