"""Shared fixtures: one freshly generated twin per session.

Generated rather than read from sim/twin.xml, which may predate the current
CAD. Generation takes about a minute and a half, so it happens once.
"""

from __future__ import annotations

import pytest

from robotic_arm.reference import REFERENCE_DIR


@pytest.fixture(scope="session")
def twin_path(tmp_path_factory):
    if not (REFERENCE_DIR / "mjcf" / "assets").exists():
        pytest.skip("run: uv run python scripts/fetch_reference.py")
    from learning.scene import ensure_twin

    return ensure_twin(tmp_path_factory.mktemp("learning") / "twin.xml", regenerate=True)


@pytest.fixture(scope="session")
def blind_env(twin_path):
    """The env without cameras: everything but rendering, and fast."""
    from learning.env import PickCubeEnv

    env = PickCubeEnv(twin_path=twin_path, cameras=())
    yield env
    env.close()


@pytest.fixture(scope="session")
def env(twin_path):
    from learning.env import PickCubeEnv

    env = PickCubeEnv(twin_path=twin_path)
    yield env
    env.close()
