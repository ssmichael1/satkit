"""Tests for the round-2 TLE / sgp4 / OMM additions of the Python API review."""

import pickle

import numpy as np
import pytest

import satkit as sk

ISS_NAME = "0 ISS (ZARYA)"
ISS = [
    "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9996",
    "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302874",
]

# Every keyword of the TLE constructor, with a value unlike the empty TLE's
KWARGS = {
    "name": "TEST SAT",
    "satnum": 43210,
    "intl_desig": "18099BC",
    "classification": "S",
    "epoch": sk.time(2024, 3, 5, 6, 7, 8.25),
    "inclination": 97.4,
    "raan": 123.25,
    "eccen": 0.0012345,
    "arg_of_perigee": 270.5,
    "mean_anomaly": 89.75,
    "mean_motion": 15.2,
    "mean_motion_dot": 1.5e-5,
    "mean_motion_dot_dot": 0.0,
    "bstar": 2.5e-4,
    "ephem_type": 0,
    "element_num": 999,
    "rev_num": 12345,
}


def _iss() -> sk.TLE:
    return sk.TLE.from_lines([ISS_NAME, *ISS])[0]


class TestKeywordConstructor:
    def test_no_arguments_is_the_empty_tle(self):
        tle = sk.TLE()
        assert (tle.name, tle.satnum, tle.intl_desig, tle.classification) == ("none", 0, "", "U")
        assert tle.epoch == sk.time.from_jd(2451545.0, sk.timescale.TT)
        for attr in ("inclination", "raan", "eccen", "arg_of_perigee", "mean_anomaly",
                     "mean_motion", "mean_motion_dot", "mean_motion_dot_dot", "bstar"):
            assert getattr(tle, attr) == 0.0
        assert (tle.ephem_type, tle.element_num, tle.rev_num) == (0, 0, 0)
        assert sk.TLE() == tle

    def test_every_keyword_sets_its_property(self):
        tle = sk.TLE(**KWARGS)
        for key, value in KWARGS.items():
            assert getattr(tle, key) == value, key

    def test_omitted_and_none_keep_the_defaults(self):
        empty = sk.TLE()
        tle = sk.TLE(satnum=5, name=None, epoch=None)
        assert tle.satnum == 5
        assert tle.name == empty.name and tle.epoch == empty.epoch

    def test_matches_setting_the_properties(self):
        built = sk.TLE()
        for key, value in KWARGS.items():
            setattr(built, key, value)
        assert sk.TLE(**KWARGS) == built

    def test_to_2line_round_trip(self):
        tle = sk.TLE(**KWARGS)
        back = sk.TLE.from_lines([tle.name, *tle.to_2line()])[0]
        # Written as text, so equal to the format's precision
        for key in ("name", "satnum", "intl_desig", "classification", "ephem_type",
                    "element_num", "rev_num"):
            assert getattr(back, key) == getattr(tle, key), key
        for key in ("inclination", "raan", "eccen", "arg_of_perigee", "mean_anomaly",
                    "mean_motion", "mean_motion_dot", "bstar"):
            assert getattr(back, key) == pytest.approx(getattr(tle, key), rel=1e-12), key
        assert abs((back.epoch - tle.epoch).seconds) < 1e-3
        assert back.to_2line() == tle.to_2line()
        # Built again from the parsed values, it is equal to the parsed TLE
        assert sk.TLE(**{k: getattr(back, k) for k in KWARGS}) == back
        # and an element set read from lines rebuilds exactly from keywords
        iss = _iss()
        rebuilt = sk.TLE(**{k: getattr(iss, k) for k in KWARGS})
        assert rebuilt == iss
        assert rebuilt.to_3line() == [ISS_NAME[2:], *ISS]

    def test_datetime_epoch(self):
        import datetime

        dt = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
        assert sk.TLE(epoch=dt).epoch == sk.time(2024, 1, 2, 3, 4, 5)
        assert sk.TLE(epoch=np.datetime64("2024-01-02T03:04:05")).epoch == sk.time(2024, 1, 2, 3, 4, 5)

    @pytest.mark.parametrize(
        "key,value,exc",
        [
            ("satnum", -1, ValueError),
            ("satnum", 1_000_000_000, ValueError),
            ("satnum", 2**40, OverflowError),
            ("satnum", "1", TypeError),
            ("eccen", 1.0, ValueError),
            ("eccen", -0.1, ValueError),
            ("eccen", float("nan"), ValueError),
            ("inclination", float("inf"), ValueError),
            ("raan", float("nan"), ValueError),
            ("arg_of_perigee", float("nan"), ValueError),
            ("mean_anomaly", float("nan"), ValueError),
            ("mean_motion", -1.0, ValueError),
            ("mean_motion_dot", float("nan"), ValueError),
            ("mean_motion_dot_dot", float("nan"), ValueError),
            ("bstar", float("inf"), ValueError),
            ("intl_desig", "123456789", ValueError),
            ("classification", "UU", ValueError),
            ("classification", "1", ValueError),
            ("ephem_type", 256, OverflowError),
            ("epoch", [sk.time(2024, 1, 1)], TypeError),
            ("epoch", "2024-01-01", TypeError),
        ],
    )
    def test_validated_as_the_setter(self, key, value, exc):
        with pytest.raises(exc) as from_kwarg:
            sk.TLE(**{key: value})
        with pytest.raises(exc) as from_setter:
            setattr(sk.TLE(), key, value)
        if exc is ValueError:
            assert str(from_kwarg.value) == str(from_setter.value)

    def test_keywords_only(self):
        with pytest.raises(TypeError):
            sk.TLE(25544)  # type: ignore[misc]
        with pytest.raises(TypeError):
            sk.TLE(desig_year=98)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            sk.TLE(sat_num=1)  # type: ignore[call-arg]

    def test_pickle_and_sgp4(self):
        tle = sk.TLE(**KWARGS)
        assert pickle.loads(pickle.dumps(tle)) == tle
        pos, vel = sk.sgp4(tle, tle.epoch)
        assert np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))


class TestFitIdentity:
    @staticmethod
    def _states():
        tle = _iss()
        times = [tle.epoch + sk.duration(minutes=10 * i) for i in range(12)]
        pos, vel = sk.sgp4(tle, times)
        q = sk.frametransform.rotation(sk.frame.TEME, sk.frame.GCRF, times)
        states = [np.hstack((qi * p, qi * v)) for qi, p, v in zip(q, pos, vel)]
        return tle, states, times

    def test_identity_keywords(self):
        tle, states, times = self._states()
        plain, _ = sk.TLE.fit_from_states(states, times, tle.epoch)
        assert (plain.name, plain.satnum, plain.intl_desig) == ("none", 0, "")
        named, _ = sk.TLE.fit_from_states(
            states, times, tle.epoch, satnum=25544, name="ISS (ZARYA)", intl_desig=" 98067A "
        )
        assert (named.name, named.satnum, named.intl_desig) == ("ISS (ZARYA)", 25544, "98067A")
        # Only the identity differs
        named.name, named.satnum, named.intl_desig = "none", 0, ""
        assert named == plain
        only, _ = sk.TLE.fit_from_states(states, times, tle.epoch, satnum=7)
        assert (only.name, only.satnum, only.intl_desig) == ("none", 7, "")

    def test_identity_checked_before_the_fit(self):
        tle = _iss()
        # Empty states would fail the fit itself; the identity check comes first
        with pytest.raises(ValueError, match="satnum"):
            sk.TLE.fit_from_states([], [], tle.epoch, satnum=-1)
        with pytest.raises(ValueError, match="intl_desig"):
            sk.TLE.fit_from_states([], [], tle.epoch, intl_desig="123456789")
        with pytest.raises(TypeError):
            sk.TLE.fit_from_states([], [], tle.epoch, name=3)  # type: ignore[arg-type]


class TestEccentricityAlias:
    def test_getter_and_setter(self):
        tle = _iss()
        assert tle.eccentricity == tle.eccen == 0.0001458
        tle.eccentricity = 0.25
        assert tle.eccen == 0.25
        tle.eccen = 0.5
        assert tle.eccentricity == 0.5

    def test_validated(self):
        tle = _iss()
        for bad in (1.0, -1e-9, float("nan")):
            with pytest.raises(ValueError):
                tle.eccentricity = bad
        assert tle.eccentricity == 0.0001458


class TestEnumFromInt:
    @pytest.mark.parametrize(
        "cls,members",
        [
            (sk.sgp4_gravconst, [sk.sgp4_gravconst.wgs72, sk.sgp4_gravconst.wgs72old,
                                 sk.sgp4_gravconst.wgs84]),
            (sk.sgp4_opsmode, [sk.sgp4_opsmode.afspc, sk.sgp4_opsmode.improved]),
            (sk.tlefitstatus, [sk.tlefitstatus.GradientConverged, sk.tlefitstatus.StepConverged,
                               sk.tlefitstatus.CostConverged, sk.tlefitstatus.MaxIterations,
                               sk.tlefitstatus.DampingSaturated]),
        ],
    )
    def test_round_trip_and_range(self, cls, members):
        for m in members:
            assert cls(int(m)) is m or cls(int(m)) == m
            assert cls(int(m)) == m
            assert pickle.loads(pickle.dumps(cls(int(m)))) == m
        for bad in (-1, len(members)):
            with pytest.raises(ValueError, match=cls.__name__):
                cls(bad)
        with pytest.raises(TypeError):
            cls("wgs72")

    def test_usable_as_sgp4_arguments(self):
        tle = _iss()
        a = sk.sgp4(tle, tle.epoch, gravconst=sk.sgp4_gravconst(2), opsmode=sk.sgp4_opsmode(1))
        b = sk.sgp4(tle, tle.epoch, gravconst=sk.sgp4_gravconst.wgs84,
                    opsmode=sk.sgp4_opsmode.improved)
        assert np.array_equal(a[0], b[0])


class TestSgp4ErrorReporting:
    """The sgp4 docstring's account of per-element-set errors (item 5)."""

    def test_init_failure_in_a_list(self):
        good = _iss()
        bad = _iss()
        bad.mean_motion = 0.0
        with pytest.raises(RuntimeError, match="element set 1"):
            sk.sgp4([good, bad], good.epoch)
        pos, vel, err = sk.sgp4([good, bad], [good.epoch, good.epoch], errflag=True)
        assert pos.shape == (2, 2, 3) and err.shape == (2, 2)
        assert np.all(np.isfinite(pos[0])) and np.all(np.isnan(pos[1]))
        assert np.all(err[0] == sk.sgp4_error.success)
        assert np.all(err[1] == err[1][0]) and err[1][0] != 0
        assert sk.sgp4_error(int(err[1][0])) != sk.sgp4_error.success

    def test_failure_at_a_time_never_raises(self):
        tle = _iss()
        tle.bstar = 0.5  # decays within days
        times = [tle.epoch + sk.duration(days=d) for d in (0, 300)]
        pos, _ = sk.sgp4(tle, times)
        assert np.all(np.isfinite(pos[0])) and np.all(np.isnan(pos[1]))
        _, _, err = sk.sgp4(tle, times, errflag=True)
        assert err[0] == sk.sgp4_error.success and err[1] != sk.sgp4_error.success

    def test_single_scalar_error_shape(self):
        tle = _iss()
        _, _, err = sk.sgp4(tle, tle.epoch, errflag=True)
        assert err.shape == (1,) and err.dtype == np.int32


class TestOmmDesignatorWithoutPiece:
    def test_round_trip(self):
        tle = _iss()
        tle.intl_desig = "98067"
        omm = tle.to_omm()
        assert omm["OBJECT_ID"] == "1998-067"
        back = sk.TLE.from_omm(omm)
        assert back.intl_desig == "98067"
        assert back == tle

    def test_with_piece_unchanged(self):
        tle = _iss()
        assert tle.to_omm()["OBJECT_ID"] == "1998-067A"
        assert sk.TLE.from_omm(tle.to_omm()).intl_desig == "98067A"
