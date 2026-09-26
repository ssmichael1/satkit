"""
satkit's warnings reach Python's ``logging`` module, as loggers under
``satkit`` (``satkit.<rust module>``).

The warning used in-process is the one for an unusable ``SATKIT_CA_BUNDLE``:
it is logged on every download attempt (most satkit warnings are once per
process), and a download from a closed local port fails at once without
touching the network. The once-per-process warnings are checked in fresh
interpreters.
"""

import logging
import os
import subprocess
import sys
import threading

import pytest

import satkit as sk

# Nothing listens on the discard port locally: the request fails at once.
CLOSED_URL = "http://127.0.0.1:9/satkit-logging-test"
CA_WARNING = "ignoring SATKIT_CA_BUNDLE="


@pytest.fixture
def bad_ca_bundle(monkeypatch, tmp_path):
    """Online, with SATKIT_CA_BUNDLE naming a missing file."""
    monkeypatch.setenv("SATKIT_CA_BUNDLE", str(tmp_path / "missing.pem"))
    was = sk.utils.is_offline()
    sk.utils.set_offline(False)
    try:
        yield
    finally:
        sk.utils.set_offline(was)


def _fetch():
    with pytest.raises(RuntimeError):
        sk.TLE.from_url(CLOSED_URL)


def _ca_records(caplog):
    return [r for r in caplog.records if CA_WARNING in r.getMessage()]


def test_warning_goes_to_satkit_logger(bad_ca_bundle, caplog):
    with caplog.at_level(logging.WARNING):
        _fetch()
    recs = _ca_records(caplog)
    assert len(recs) == 1, caplog.text
    assert recs[0].name == "satkit.utils.download"
    assert recs[0].levelno == logging.WARNING
    assert "platform trust store" in recs[0].getMessage()


def test_setlevel_after_first_record_silences(bad_ca_bundle, caplog):
    # A record has already gone through this logger (levels are not cached
    # on the Rust side, so a later setLevel still applies).
    with caplog.at_level(logging.WARNING):
        _fetch()
        assert _ca_records(caplog)
        caplog.clear()
        logger = logging.getLogger("satkit")
        old = logger.level
        logger.setLevel(logging.ERROR)
        try:
            _fetch()
        finally:
            logger.setLevel(old)
        assert not _ca_records(caplog), caplog.text
        _fetch()
        assert len(_ca_records(caplog)) == 1


def test_warning_from_background_thread(bad_ca_bundle, caplog):
    # The fetch releases the GIL; the warning re-acquires it to log.
    errors = []

    def work():
        try:
            sk.TLE.from_url(CLOSED_URL)
        except RuntimeError as e:
            errors.append(e)

    with caplog.at_level(logging.WARNING):
        th = threading.Thread(target=work, name="satkit-log-bg")
        th.start()
        th.join(60)
        assert not th.is_alive(), "logging from a background thread deadlocked"
    assert len(errors) == 1
    recs = _ca_records(caplog)
    assert len(recs) == 1, caplog.text
    assert recs[0].threadName == "satkit-log-bg"


def _run(code, env=None, timeout=120):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ, **(env or {})),
    )


EOP_1950 = "import satkit as sk; sk.frametransform.earth_orientation_params(sk.time(1950, 1, 1))"


@pytest.mark.parametrize(
    "setup, shown",
    [
        # Unconfigured logging: WARNING records reach stderr through
        # logging's last-resort handler, as they did before.
        ("", True),
        ("import logging; logging.getLogger('satkit').setLevel(logging.ERROR)", False),
        ("import logging; logging.getLogger('satkit.earth_orientation_params').disabled = True", False),
        ("import satkit; satkit.frametransform.disable_eop_time_warning()", False),
    ],
)
def test_one_time_eop_warning_default_visibility(setup, shown):
    # Loaded table: the epoch is before it. No table: the no-EOP warning.
    # Either names EOP.
    out = _run(f"{setup}\n{EOP_1950}")
    assert out.returncode == 0, out.stderr
    assert ("EOP" in out.stderr) == shown, out.stderr


def test_handler_receives_one_time_warning():
    out = _run(
        "import logging, sys\n"
        "logging.basicConfig(stream=sys.stdout, format='%(levelname)s %(name)s: %(message)s')\n"
        + EOP_1950
    )
    assert out.returncode == 0, out.stderr
    assert "WARNING satkit.earth_orientation_params: " in out.stdout, out.stdout
    assert "EOP" not in out.stderr, out.stderr


def test_update_datafiles_worker_warnings_do_not_deadlock(tmp_path):
    # update_datafiles downloads on worker threads and joins them; each
    # worker logs the CA-bundle warning, which needs the GIL. Every request
    # goes to a closed local proxy, so all of them fail at once.
    proxy = "http://127.0.0.1:9"
    env = {
        "SATKIT_CA_BUNDLE": str(tmp_path / "missing.pem"),
        "SATKIT_DATA": str(tmp_path),
        "SATKIT_OFFLINE": "0",
        "HTTPS_PROXY": proxy,
        "HTTP_PROXY": proxy,
        "ALL_PROXY": proxy,
        "https_proxy": proxy,
        "http_proxy": proxy,
        "all_proxy": proxy,
        "NO_PROXY": "",
        "no_proxy": "",
    }
    code = (
        "import satkit as sk\n"
        f"try:\n    sk.utils.update_datafiles(dir={str(tmp_path)!r})\n"
        "except RuntimeError as e:\n    print('failed as expected')\n"
    )
    out = _run(code, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "failed as expected" in out.stdout, out.stdout
    assert CA_WARNING in out.stderr, out.stderr
