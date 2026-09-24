"""Space-weather coverage, provenance and status (issue #202)."""

import satkit as sk


def test_coverage_bounds():
    cov = sk.spaceweather.coverage()
    assert cov is not None, "the space-weather files must be available for the test suite"
    first, last_observed, last_daily, last = cov
    assert isinstance(first, sk.time)
    # The GFZ record starts in 1932 (CelesTrak's in 1957) and the boundaries
    # are ordered: observed <= daily <= end of table.
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
    assert sk.spaceweather.status(first - sk.duration.from_days(1)) == "before_table"
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
    # the newest GFZ rows are nowcast values, flagged preliminary
    assert sk.spaceweather.get(last_observed)["data_type"] in ("OBS", "OBS-P", "INT")
    if last_daily < last:
        monthly = sk.spaceweather.get(last_daily + sk.duration.from_days(20))
        assert monthly["data_type"] == "PRM"


def test_monthly_rows_carry_ap():
    """Issue #202: past the daily rows the table used to hold only monthly
    F10.7, so NRLMSISE-00 ran on a quiet-time Ap = 4. MSAFE monthly rows
    carry a climatological Ap, so every row has geomagnetic data."""
    _, _, last_daily, last = sk.spaceweather.coverage()
    if last_daily >= last:
        return  # no MSAFE file provisioned (offline CI): nothing to check
    t = last_daily + sk.duration.from_days(20)
    assert sk.spaceweather.status(t) == "predicted_monthly"
    rec = sk.spaceweather.get(t)
    assert rec["data_type"] == "PRM"
    assert rec["ap_avg"] >= 0
    assert all(a == rec["ap_avg"] for a in rec["ap"])
    assert rec["f10p7_obs"] > 0
    # and the 81-day averages are filled across the forecast, not sentinels
    assert rec["f10p7_obs_c81"] > 0


def _provisioned_table_path():
    """The file the default loader would read, so a test that replaces the
    singleton can put the real table back."""
    from pathlib import Path
    for name in ("Kp_ap_Ap_SN_F107_since_1932.txt", "SW-All.csv"):
        for d in sk.utils.data_search_dirs():
            f = Path(d) / name
            if f.is_file():
                return f
    return None


def test_init_from_path_round_trip(tmp_path):
    """A GFZ-format buffer loads as an observed-only table — the
    bring-your-own-file path — and the real table is restored afterwards so
    later tests are not left querying a one-row singleton."""
    restore = _provisioned_table_path()
    if restore is None:
        return
    before = sk.spaceweather.get(sk.time(2023, 3, 1))["f10p7_obs"]
    row = (
        "# GFZ-style header\n"
        "2023 03 01 33297 33297.5 2580 4  1.333  2.333  2.333  2.000  1.667  1.333  1.000  0.667"
        "    5    9    9    7    6    5    4    3     6  92  162.0  165.0 2\n"
    )
    f = tmp_path / "Kp_ap_Ap_SN_F107_since_1932.txt"
    f.write_text(row)
    try:
        sk.spaceweather.init_from_path(f)
        only = sk.spaceweather.get(sk.time(2023, 3, 1))
        assert only["f10p7_obs"] == 162.0
        assert only["data_type"] == "OBS"
        assert only["isn"] == -1  # SN is CC BY-NC and never ingested
        first, last_obs, last_daily, last = sk.spaceweather.coverage()
        assert first == last == last_obs == last_daily
    finally:
        sk.spaceweather.init_from_path(restore)
    assert sk.spaceweather.get(sk.time(2023, 3, 1))["f10p7_obs"] == before


def test_disable_warning_is_callable():
    sk.spaceweather.disable_space_weather_time_warning()
