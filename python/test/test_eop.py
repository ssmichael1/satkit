"""
Earth Orientation Parameter (EOP) coverage: the table bounds, the status of
an epoch relative to them, and the propagator's behaviour past the table end.
"""

import pickle

import numpy as np
import pytest

import satkit as sk


def test_eop_coverage_bounds():
    cov = sk.frametransform.eop_coverage()
    assert cov is not None, "EOP-All.csv must be available for the test suite"
    first, last_observed, last = cov
    assert isinstance(first, sk.time)
    assert first < last_observed <= last
    # The table starts in 1962 and must cover a well-observed historical epoch.
    assert first < sk.time(1963, 1, 1)
    assert last_observed > sk.time(2020, 1, 1)


def test_eop_status_values():
    first, last_observed, last = sk.frametransform.eop_coverage()
    assert sk.frametransform.eop_status(sk.time(2006, 4, 16, 17, 52, 50)) == "observed"
    assert sk.frametransform.eop_status(first) == "observed"
    assert sk.frametransform.eop_status(last_observed) == "observed"
    assert sk.frametransform.eop_status(sk.time(1950, 1, 1)) == "before_table"
    assert sk.frametransform.eop_status(last + sk.duration.from_days(10)) == "extrapolated"
    if last_observed < last:
        assert sk.frametransform.eop_status(last_observed + sk.duration.from_days(1)) == "predicted"
    # earth_orientation_params still answers (constant extrapolation) past the end.
    assert sk.frametransform.earth_orientation_params(last + sk.duration.from_days(10)) is not None


def _leo_state():
    return np.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])


def test_require_eop_coverage_raises_past_table_end():
    _, _, last = sk.frametransform.eop_coverage()
    t0 = last + sk.duration.from_days(30)
    t1 = t0 + sk.duration.from_seconds(600)
    strict = sk.propsettings(require_eop_coverage=True)
    assert strict.require_eop_coverage is True
    with pytest.raises(RuntimeError, match="EOP data ends"):
        sk.propagate(_leo_state(), t0, end=t1, propsettings=strict)
    # Default: extrapolates and succeeds.
    res = sk.propagate(_leo_state(), t0, end=t1, propsettings=sk.propsettings())
    assert np.all(np.isfinite(res.state))
    # Inside coverage the flag is inert.
    t0 = last - sk.duration.from_days(30)
    sk.propagate(_leo_state(), t0, end=t0 + sk.duration.from_seconds(600), propsettings=strict)


def test_require_eop_coverage_property_and_pickle():
    ps = sk.propsettings()
    assert ps.require_eop_coverage is False
    ps.require_eop_coverage = True
    restored = pickle.loads(pickle.dumps(ps))
    assert restored.require_eop_coverage is True
    assert "Require EOP Coverage: true" in str(restored)
