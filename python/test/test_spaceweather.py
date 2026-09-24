"""Space-weather coverage, provenance and status (issue #202)."""

import satkit as sk


def test_coverage_bounds():
    cov = sk.spaceweather.coverage()
    assert cov is not None, "a space-weather file (SW-All.csv) must be available for the test suite"
    first, last_observed, last_daily, last = cov
    assert isinstance(first, sk.time)
    # The CelesTrak table starts in 1957 and the boundaries are ordered:
    # observed <= daily <= end of table.
    assert first < sk.time(1958, 1, 1)
    assert first < last_observed <= last_daily <= last
    assert last_observed > sk.time(2020, 1, 1)
    # Past the daily rows the table continues for years at monthly cadence.
    assert last > last_daily


def test_status_values():
    first, last_observed, last_daily, last = sk.spaceweather.coverage()
    assert sk.spaceweather.status(sk.time(2023, 3, 1)) == "observed"
    assert sk.spaceweather.status(first) == "observed"
    assert sk.spaceweather.status(last_observed) == "observed"
    assert sk.spaceweather.status(sk.time(1950, 1, 1)) == "before_table"
    assert sk.spaceweather.status(last + sk.duration.from_days(10)) == "extrapolated"
    if last_observed < last_daily:
        assert (
            sk.spaceweather.status(last_observed + sk.duration.from_days(1))
            == "predicted_daily"
        )
    if last_daily < last:
        assert (
            sk.spaceweather.status(last_daily + sk.duration.from_days(1))
            == "predicted_monthly"
        )


def test_data_type_provenance():
    """The F10.7_DATA_TYPE column is parsed and exposed, not skipped."""
    _, last_observed, last_daily, last = sk.spaceweather.coverage()
    assert sk.spaceweather.get(sk.time(2023, 3, 1))["data_type"] in ("OBS", "INT")
    assert sk.spaceweather.get(last_observed)["data_type"] in ("OBS", "INT")
    if last_daily < last:
        monthly = sk.spaceweather.get(last_daily + sk.duration.from_days(20))
        assert monthly["data_type"] == "PRM"


def test_monthly_rows_carry_no_geomagnetic_data():
    """The defect behind the Ap=4 fallback: monthly rows have -1 for every
    Kp/ap field, and `status` is what tells a caller before propagating."""
    _, _, last_daily, last = sk.spaceweather.coverage()
    if last_daily >= last:
        return
    t = last_daily + sk.duration.from_days(20)
    assert sk.spaceweather.status(t) == "predicted_monthly"
    rec = sk.spaceweather.get(t)
    assert rec["ap_avg"] == -1
    assert all(a == -1 for a in rec["ap"])
    # F10.7 is still published for these rows.
    assert rec["f10p7_obs"] > 0


def test_disable_warning_is_callable():
    sk.spaceweather.disable_space_weather_time_warning()
