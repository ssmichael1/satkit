"""The deprecated ``as_X`` conversion aliases on ``time`` and ``quaternion``.

Renamed to ``to_X`` in 0.23 (paired with the ``from_X`` constructors); the
old names warn and are removed in 0.25.  Each alias must emit a
``DeprecationWarning`` naming its replacement and return the same value.
"""

import os
import warnings

import numpy as np
import pytest

import satkit as sk


T = sk.time(2023, 6, 3, 6, 19, 34.25)
Q = sk.quaternion.rotz(0.3) * sk.quaternion.roty(-0.2) * sk.quaternion.rotx(0.1)

# (object, old name, new name, positional args, keyword args)
CASES = [
    (T, "as_date", "to_date", (), {}),
    (T, "as_gregorian", "to_gregorian", (), {}),
    (T, "as_datetime", "to_datetime", (), {}),
    (T, "as_datetime", "to_datetime", (), {"utc": True}),
    (T, "as_datetime", "to_datetime", (False,), {}),
    (T, "as_mjd", "to_mjd", (), {}),
    (T, "as_mjd", "to_mjd", (sk.timescale.TT,), {}),
    (T, "as_mjd", "to_mjd", (), {"scale": sk.timescale.TAI}),
    (T, "as_jd", "to_jd", (), {}),
    (T, "as_jd", "to_jd", (sk.timescale.GPS,), {}),
    (T, "as_unixtime", "to_unixtime", (), {}),
    (T, "as_iso8601", "to_iso8601", (), {}),
    (T, "as_rfc3339", "to_rfc3339", (), {}),
    (Q, "as_rotation_matrix", "to_rotation_matrix", (), {}),
    (Q, "as_euler", "to_euler", (), {}),
]


def _ids():
    out = []
    for _, old, _, args, kwargs in CASES:
        parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        out.append(f"{old}({', '.join(parts)})")
    return out


@pytest.mark.parametrize("obj, old, new, args, kwargs", CASES, ids=_ids())
def test_alias_warns_and_matches(obj, old, new, args, kwargs):
    cls = type(obj).__name__
    with pytest.warns(
        DeprecationWarning,
        match=rf"{cls}\.{old}\(\) is deprecated since 0\.23 .* use {cls}\.{new}\(\)",
    ):
        got = getattr(obj, old)(*args, **kwargs)
    expected = getattr(obj, new)(*args, **kwargs)
    if isinstance(got, np.ndarray):
        np.testing.assert_array_equal(got, expected)
    else:
        assert got == expected


@pytest.mark.parametrize("obj, old, new, args, kwargs", CASES, ids=_ids())
def test_new_name_does_not_warn(obj, old, new, args, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(obj, new)(*args, **kwargs)


def test_warning_points_at_caller():
    """stacklevel 2: the warning is attributed to this file, not the binding."""
    with pytest.warns(DeprecationWarning) as record:
        T.as_mjd()
    assert len(record) == 1
    assert record[0].filename == __file__


def _kepler():
    return sk.kepler(7000e3, 0.01, 0.9, 0.3, 0.2, 0.1)


def test_kepler_w_warning_points_at_caller():
    """kepler.w (getter and setter) warns from the caller's line."""
    k = _kepler()
    with pytest.warns(DeprecationWarning, match="kepler.argp") as record:
        w = k.w
    assert w == k.argp
    assert len(record) == 1
    assert record[0].filename == __file__
    with pytest.warns(DeprecationWarning) as record:
        k.w = 0.5
    assert k.argp == 0.5
    assert record[0].filename == __file__


def test_kepler_w_warning_shown_by_default_in_main():
    """Attributed to ``__main__``, the default filters show it (they hide a
    DeprecationWarning attributed to ``<sys>``)."""
    import subprocess
    import sys

    code = "import satkit as sk; k = sk.kepler(7000e3, 0.01, 0.9, 0.3, 0.2, 0.1); k.w"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert "DeprecationWarning: kepler.w is deprecated" in out.stderr
    assert "<string>:1" in out.stderr
