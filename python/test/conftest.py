import os

import pytest


@pytest.fixture
def testvec_dir():
    return os.getenv(
        "SATKIT_TESTVEC_ROOT",
        default="." + os.path.sep + "satkit-testvecs" + os.path.sep,
    )
