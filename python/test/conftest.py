import os

import pytest


@pytest.fixture
def testvec_dir():
    return os.getenv(
        "SATKIT_TESTVEC_ROOT",
        default="." + os.path.sep + "satkit-testvecs" + os.path.sep,
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """With no EOP table at all (no finals2000A.all in the data directories and
    no network, e.g. a directory holding only CelesTrak's EOP-All.csv, which
    satkit does not read), the EOP-dependent tests cannot run: report them as
    skipped, not failed. With a table loaded this changes nothing."""
    try:
        return (yield)
    except (Exception, pytest.fail.Exception):
        import satkit as sk

        if sk.frametransform.eop_coverage() is None:
            pytest.skip("no EOP table loaded; run satkit.utils.update_datafiles() to fetch finals2000A.all")
        raise
