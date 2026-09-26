import os

import pytest

# Hypothesis profiles, selected with HYPOTHESIS_PROFILE (default: hypothesis's
# own "default", random). Pull-request CI uses "ci" so a PR's result does not
# depend on the random draw; the weekly scheduled job uses "deep" (random,
# with HYPOTHESIS_MAX_EXAMPLES raised; see python/test/test_properties.py).
# Per-test @settings(...) set max_examples/deadline only, so the profile's
# seeding applies to every property.
try:
    from hypothesis import settings as _hsettings

    _hsettings.register_profile("ci", derandomize=True, database=None, print_blob=True)
    _hsettings.register_profile("deep", derandomize=False, database=None, print_blob=True)
    _hsettings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
except ImportError:  # hypothesis is optional; test_properties.py skips without it
    pass


@pytest.fixture
def testvec_dir():
    return os.getenv(
        "SATKIT_TESTVEC_ROOT",
        default="." + os.path.sep + "satkit-testvecs" + os.path.sep,
    )
