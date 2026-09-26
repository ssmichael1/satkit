"""NRLMSISE-00 density bindings: ``satkit.nrlmsise00`` and ``satkit.density.nrlmsise``."""

import datetime
import math

import pytest

import satkit as sk


def test_density_itrfcoord_matches_degree_call():
    # density.nrlmsise(itrfcoord, time) must hand the model geodetic latitude
    # and longitude in degrees, like nrlmsise00(..., latitude_deg, longitude_deg).
    t = sk.time(2023, 3, 1, 12, 0, 0)
    coord = sk.itrfcoord(latitude_deg=60.0, longitude_deg=-70.0, altitude=400e3)
    rho_a, temp_a = sk.density.nrlmsise(coord, t)
    rho_b, temp_b = sk.nrlmsise00(
        coord.altitude / 1e3,
        latitude_deg=coord.latitude_deg,
        longitude_deg=coord.longitude_deg,
        time=t,
    )
    assert math.isclose(rho_a, rho_b, rel_tol=1e-9)
    assert math.isclose(temp_a, temp_b, rel_tol=1e-9)
    # The radians interpretation (lat ~ 1.05 deg) is a different atmosphere.
    rho_rad, _ = sk.nrlmsise00(
        400.0,
        latitude_deg=math.radians(60.0),
        longitude_deg=math.radians(-70.0),
        time=t,
    )
    assert abs(rho_rad / rho_a - 1.0) > 1e-3


def test_spaceweather_feed_changes_density_within_the_day():
    # 2023-02-27 is a G2 storm day (3-hourly ap 48 -> 111 in the morning, daily
    # Ap 91); with the 3-hourly history the density at the same place moves
    # between 01:30 and 13:30 UT beyond what the local-time cycle alone gives.
    coord = dict(latitude_deg=0.0, longitude_deg=0.0)
    rho_storm = [
        sk.nrlmsise00(420.0, time=sk.time(2023, 2, 27, h, 30, 0), **coord)[0]
        for h in (1, 13)
    ]
    rho_quiet = [
        sk.nrlmsise00(420.0, time=sk.time(2023, 3, 1, h, 30, 0), **coord)[0]
        for h in (1, 13)
    ]
    for r in rho_storm + rho_quiet:
        assert r > 0.0
    # Storm day: the 13:30 UT density (after the 111 intervals) exceeds the
    # 01:30 UT density by clearly more than the quiet day's local-time ratio
    # (measured: 2.23 vs 1.81).
    assert rho_storm[1] / rho_storm[0] > 1.1 * rho_quiet[1] / rho_quiet[0]
    # Fixed indices ignore the file entirely.
    rho_fixed = sk.nrlmsise00(420.0, time=sk.time(2023, 2, 27, 13, 30, 0), use_spaceweather=False, **coord)[0]
    assert abs(rho_fixed / rho_storm[1] - 1.0) > 0.05


def test_density_float_form_takes_radians():
    # density.nrlmsise(altitude_m, latitude_rad, longitude_rad, time) is
    # documented in radians; it must land on the same atmosphere as the
    # itrfcoord form (which hands the model degrees) for the same point.
    t = sk.time(2023, 3, 1, 12, 0, 0)
    coord = sk.itrfcoord(latitude_deg=60.0, longitude_deg=-70.0, altitude=400e3)
    rho_coord, temp_coord = sk.density.nrlmsise(coord, t)
    rho_float, temp_float = sk.density.nrlmsise(
        400e3, math.radians(60.0), math.radians(-70.0), t
    )
    assert math.isclose(rho_float, rho_coord, rel_tol=1e-6)
    assert math.isclose(temp_float, temp_coord, rel_tol=1e-6)


def test_density_accepts_datetime_and_rejects_unknown_time():
    # density.nrlmsise takes the same single-time types as the rest of the
    # API; anything else is a TypeError, not a silent "no time" (which runs
    # the model on its defaults, 8x low in the 2024-05-11 storm).
    t = sk.time(2024, 5, 11, 12, 0, 0)
    dt = datetime.datetime(2024, 5, 11, 12, 0, 0, tzinfo=datetime.timezone.utc)
    coord = sk.itrfcoord(latitude_deg=30.0, longitude_deg=40.0, altitude=400e3)
    ref = sk.density.nrlmsise(coord, t)
    assert ref != sk.density.nrlmsise(coord)
    assert sk.density.nrlmsise(coord, None) == sk.density.nrlmsise(coord)
    assert sk.density.nrlmsise(coord, dt) == ref
    lat, lon = math.radians(30.0), math.radians(40.0)
    assert sk.density.nrlmsise(400e3, lat, lon, dt) == sk.density.nrlmsise(400e3, lat, lon, t)
    for bad in ("2024-05-11", 12345, object()):
        with pytest.raises(TypeError):
            sk.density.nrlmsise(coord, bad)
        with pytest.raises(TypeError):
            sk.density.nrlmsise(400e3, lat, lon, bad)
    with pytest.raises(TypeError):
        sk.density.nrlmsise(400e3, "north", lon, t)
    with pytest.raises(TypeError):
        sk.density.nrlmsise(coord, t, t)


def test_density_integer_angles_are_not_dropped():
    # An int latitude / longitude / altitude is a real number, not "absent".
    t = sk.time(2023, 3, 1, 12, 0, 0)
    assert sk.density.nrlmsise(400e3, 1, 2, t) == sk.density.nrlmsise(400e3, 1.0, 2.0, t)
    assert sk.density.nrlmsise(400000, 1.0, 2.0, t) == sk.density.nrlmsise(400e3, 1.0, 2.0, t)
    assert sk.density.nrlmsise(400e3, 1, 2, t) != sk.density.nrlmsise(400e3, 0.0, 0.0, t)
    # The time may follow fewer angles; missing angles are 0.
    assert sk.density.nrlmsise(400e3, t) == sk.density.nrlmsise(400e3, 0.0, 0.0, t)
    assert sk.density.nrlmsise(400e3, 1.0, t) == sk.density.nrlmsise(400e3, 1.0, 0.0, t)
