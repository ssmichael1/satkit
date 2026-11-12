"""Regression tests for the TLE / sgp4 / OMM items of the Python API review."""

import pickle
import warnings

import numpy as np
import pytest

import satkit as sk

ISS_NAME = "0 ISS (ZARYA)"
ISS = [
    "1 25544U 98067A   21264.51782528  .00002893  00000-0  58680-4 0  9996",
    "2 25544  51.6442 208.5856 0001458  47.2277  50.1624 15.48919419302874",
]


def _line1(desig: str) -> str:
    """ISS line 1 with another international designator (columns 10-17)"""
    return ISS[0][:9] + f"{desig:<8}" + ISS[0][17:]


def _iss(named: bool = False) -> sk.TLE:
    return sk.TLE.from_lines([ISS_NAME, *ISS] if named else ISS)[0]


class TestDesignator:
    """Bug 7: ``intl_desig`` is the stored designator and ``desig_year``,
    ``desig_launch`` and ``desig_piece`` are its parts, in both directions."""

    def test_parts_are_read_from_the_designator(self):
        tle = _iss()
        assert tle.intl_desig == "98067A"
        assert (tle.desig_year, tle.desig_launch, tle.desig_piece) == (98, 67, "A")
        tle.intl_desig = "24001BC"
        assert (tle.desig_year, tle.desig_launch, tle.desig_piece) == (24, 1, "BC")

    def test_part_setters_reach_every_output(self):
        tle = _iss()
        tle.desig_year = 5
        assert tle.intl_desig == "05067A"
        tle.desig_launch = 123
        assert tle.intl_desig == "05123A"
        tle.desig_piece = "XYZ"
        assert tle.intl_desig == "05123XYZ"
        assert (tle.desig_year, tle.desig_launch, tle.desig_piece) == (5, 123, "XYZ")
        assert tle.to_2line()[0][9:17] == "05123XYZ"
        assert tle.to_3line()[1][9:17] == "05123XYZ"
        assert tle.to_omm()["OBJECT_ID"] == "2005-123XYZ"
        assert sk.TLE.from_lines(tle.to_2line())[0] == tle

    def test_blank_designator_has_no_year(self):
        # The year used to read as 70
        for tle in (sk.TLE(), sk.TLE.from_lines([_line1(""), ISS[1]])[0]):
            assert tle.intl_desig == ""
            assert tle.desig_year is None
            assert tle.desig_launch is None
            assert tle.desig_piece == ""
        assert tle.to_2line()[0][9:17] == " " * 8
        assert tle.to_omm()["OBJECT_ID"] == "UNKNOWN"

    def test_setting_a_part_of_a_blank_designator_zero_fills(self):
        tle = sk.TLE()
        tle.desig_piece = "B"
        assert tle.intl_desig == "00000B"
        tle = sk.TLE()
        tle.desig_year = 98
        assert tle.intl_desig == "98000"
        assert (tle.desig_year, tle.desig_launch, tle.desig_piece) == (98, 0, "")
        tle.intl_desig = ""
        assert tle.desig_year is None

    def test_three_letter_piece_round_trips(self):
        # intl_desig used to keep only the first two letters
        line1 = _line1("98067ABC")
        tle = sk.TLE.from_lines([line1, ISS[1]])[0]
        assert tle.intl_desig == "98067ABC"
        assert tle.desig_piece == "ABC"
        assert tle.to_2line()[0][:68] == line1[:68]
        assert tle.to_omm()["OBJECT_ID"] == "1998-067ABC"

    def test_designator_that_is_not_year_and_launch(self):
        tle = _iss()
        tle.intl_desig = "ANALYST"
        assert tle.desig_year is None and tle.desig_launch is None
        assert tle.desig_piece == "ST"
        assert tle.to_2line()[0][9:17] == "ANALYST "

    def test_setters_validate(self):
        tle = _iss()
        for name, bad in [
            ("desig_year", -1),
            ("desig_year", 100),
            ("desig_year", 1998),
            ("desig_launch", -1),
            ("desig_launch", 1000),
            ("desig_piece", "ABCD"),
            ("desig_piece", "1"),
            ("desig_piece", "é"),
            ("intl_desig", "1998-067A"),
            ("intl_desig", "98067é"),
        ]:
            with pytest.raises(ValueError, match=name):
                setattr(tle, name, bad)
        with pytest.raises(TypeError):
            tle.desig_year = None
        # Nothing was changed by the refused values
        assert tle == _iss()
        tle.intl_desig = "  98067B "
        assert tle.intl_desig == "98067B"

    def test_equality_and_pickle_follow_the_designator(self):
        a, b = _iss(), _iss()
        b.desig_piece = "B"
        assert a != b
        b.intl_desig = "98067A"
        assert a == b
        c = sk.TLE.from_lines([_line1("98067ABC"), ISS[1]])[0]
        d = pickle.loads(pickle.dumps(c))
        assert d == c and d.intl_desig == "98067ABC" and d.desig_piece == "ABC"
        e = pickle.loads(pickle.dumps(sk.TLE()))
        assert e == sk.TLE() and e.desig_year is None


class TestTo3Line:
    """Bug 9: ``to_3line`` raises what ``to_2line`` raises."""

    def test_satnum_without_alpha5_form(self):
        tle = _iss(named=True)
        tle.satnum = 340000
        with pytest.raises(ValueError, match="OMM") as two:
            tle.to_2line()
        with pytest.raises(ValueError, match="OMM") as three:
            tle.to_3line()
        assert str(two.value) == str(three.value)

    def test_same_lines(self):
        tle = _iss(named=True)
        assert tle.to_3line() == ["ISS (ZARYA)", *tle.to_2line()]
        assert tle.to_2line() == ISS
        assert _iss().to_3line()[0] == "none"


def _xml_dict() -> dict:
    """What ``xmltodict.parse`` returns for a CCSDS OMM XML document (built by
    hand: xmltodict is not a dependency)."""
    return {
        "omm": {
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@id": "CCSDS_OMM_VERS",
            "@version": "2.0",
            "header": {"CREATION_DATE": "2021-09-21T13:00:00", "ORIGINATOR": "TEST"},
            "body": {
                "segment": {
                    "metadata": {
                        "OBJECT_NAME": "ISS (ZARYA)",
                        "OBJECT_ID": "1998-067A",
                        "CENTER_NAME": "EARTH",
                        "REF_FRAME": "TEME",
                        "TIME_SYSTEM": "UTC",
                        "MEAN_ELEMENT_THEORY": "SGP4",
                    },
                    "data": {
                        "meanElements": {
                            "EPOCH": "2021-09-21T12:25:40.104192",
                            "MEAN_MOTION": {"@units": "rev/day", "#text": "15.48919419"},
                            "ECCENTRICITY": "0.0001458",
                            "INCLINATION": {"@units": "deg", "#text": "51.6442"},
                            "RA_OF_ASC_NODE": "208.5856",
                            "ARG_OF_PERICENTER": "47.2277",
                            "MEAN_ANOMALY": "50.1624",
                        },
                        "tleParameters": {
                            "EPHEMERIS_TYPE": "0",
                            "CLASSIFICATION_TYPE": "U",
                            "NORAD_CAT_ID": "25544",
                            "ELEMENT_SET_NO": "999",
                            "REV_AT_EPOCH": "30287",
                            "BSTAR": "0.5868E-4",
                            "MEAN_MOTION_DOT": "0.2893E-4",
                            "MEAN_MOTION_DDOT": "0",
                        },
                        "userDefinedParameters": {
                            "USER_DEFINED": {"@parameter": "NOTE", "#text": "x"}
                        },
                    },
                }
            },
        }
    }


class TestXmltodictOmm:
    """Bug 17: the dict ``xmltodict.parse`` returns is accepted whole."""

    def test_root_key_omm(self):
        t = sk.time(2021, 9, 22)
        want = sk.sgp4(_iss(), t)
        got = sk.sgp4(_xml_dict(), t)
        np.testing.assert_allclose(got[0], want[0], atol=1e-6)
        np.testing.assert_allclose(got[1], want[1], atol=1e-9)

    def test_root_key_ndm(self):
        t = sk.time(2021, 9, 22)
        want = sk.sgp4(_iss(), t)
        omm = _xml_dict()["omm"]
        for ndm in ({"ndm": {"omm": omm}}, {"ndm": {"@xmlns": "urn:x", "omm": [omm]}}):
            np.testing.assert_allclose(sk.sgp4(ndm, t)[0], want[0], atol=1e-6)

    def test_inner_nodes_still_work(self):
        t = sk.time(2021, 9, 22)
        want = sk.sgp4(_iss(), t)[0]
        d = _xml_dict()["omm"]
        for node in (d, d["body"], d["body"]["segment"], d["body"]["segment"]["data"]):
            np.testing.assert_allclose(sk.sgp4(node, t)[0], want, atol=1e-6)

    def test_from_omm(self):
        tle = sk.TLE.from_omm(_xml_dict())
        assert tle.name == "ISS (ZARYA)"
        assert tle.to_2line() == ISS

    def test_several_messages_are_refused_with_a_hint(self):
        omm = _xml_dict()["omm"]
        ndm = {"ndm": {"omm": [omm, omm]}}
        with pytest.raises(ValueError, match="holds 2 messages"):
            sk.sgp4(ndm, sk.time(2021, 9, 22))
        # ... and the hint works
        t = sk.time(2021, 9, 22)
        pos, _ = sk.sgp4([{"omm": m} for m in ndm["ndm"]["omm"]], t)
        assert pos.shape == (2, 3)


class TestOnError:
    """Bug 18: only ``on_error="raise"`` raises for what the lines contain."""

    NO_RECORDS = ([], ["garbage"], ["just a name", ""], ["1 25544U short"])
    BAD_RECORD = [ISS[0]]  # a line 1 with no line 2

    @pytest.mark.parametrize("lines", NO_RECORDS)
    def test_skip_never_raises(self, lines):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sk.TLE.from_lines(lines, on_error="skip") == []

    def test_skip_bad_record(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sk.TLE.from_lines(self.BAD_RECORD, on_error="skip") == []
            assert len(sk.TLE.from_lines([*self.BAD_RECORD, *ISS], on_error="skip")) == 1

    @pytest.mark.parametrize("lines", NO_RECORDS)
    def test_warn_says_there_were_none(self, lines):
        with pytest.warns(sk.TLEWarning, match="No valid TLEs found in input"):
            assert sk.TLE.from_lines(lines, on_error="warn") == []

    def test_warn_bad_record(self):
        with pytest.warns(sk.TLEWarning, match="Line 1 without a line 2") as rec:
            assert sk.TLE.from_lines(self.BAD_RECORD, on_error="warn") == []
        assert len(rec) == 1

    @pytest.mark.parametrize("lines", NO_RECORDS)
    def test_raise_is_the_default(self, lines):
        with pytest.raises(ValueError, match="No valid TLEs found in input"):
            sk.TLE.from_lines(lines)
        with pytest.raises(ValueError, match="No valid TLEs found in input"):
            sk.TLE.from_lines(lines, on_error="raise")

    def test_from_file_follows_the_same_rule(self, tmp_path):
        path = tmp_path / "none.tle"
        path.write_text("no element sets here\n")
        assert sk.TLE.from_file(str(path), on_error="skip") == []
        with pytest.warns(sk.TLEWarning, match="No valid TLEs found in .*none.tle"):
            assert sk.TLE.from_file(str(path), on_error="warn") == []
        with pytest.raises(ValueError, match="No valid TLEs found in .*none.tle"):
            sk.TLE.from_file(str(path))

    def test_bad_on_error_value(self):
        with pytest.raises(ValueError, match="on_error"):
            sk.TLE.from_lines(ISS, on_error="ignore")  # type: ignore[arg-type]


class TestOmmRoundTrip:
    """Bug 19: ``TLE.from_omm(t.to_omm()) == t``, unnamed TLEs included."""

    def test_unnamed(self):
        tle = _iss()
        assert tle.name == "none"
        omm = tle.to_omm()
        assert omm["OBJECT_NAME"] == "UNKNOWN"
        back = sk.TLE.from_omm(omm)
        assert back.name == "none"
        assert back == tle

    def test_named(self):
        tle = _iss(named=True)
        assert sk.TLE.from_omm(tle.to_omm()) == tle

    def test_blank_designator_and_empty_tle(self):
        blank = sk.TLE.from_lines([_line1(""), ISS[1]])[0]
        assert sk.TLE.from_omm(blank.to_omm()) == blank
        assert sk.TLE.from_omm(sk.TLE().to_omm()) == sk.TLE()

    def test_three_letter_piece(self):
        tle = sk.TLE.from_lines([_line1("98067ABC"), ISS[1]])[0]
        assert sk.TLE.from_omm(tle.to_omm()) == tle

    def test_missing_object_name_is_unnamed(self):
        omm = _iss(named=True).to_omm()
        del omm["OBJECT_NAME"]
        assert sk.TLE.from_omm(omm).name == "none"


ENUMS = [sk.sgp4_error, sk.sgp4_gravconst, sk.sgp4_opsmode, sk.tlefitstatus]


def _members(cls):
    return [v for k, v in vars(cls).items() if not k.startswith("_") and isinstance(v, cls)]


class TestEnums:
    """Review 2.1, 2.15, 4.2: hashable, frozen, in module ``satkit``."""

    @pytest.mark.parametrize("cls", ENUMS)
    def test_hashable_int_and_module(self, cls):
        members = _members(cls)
        assert len(members) >= 2
        assert cls.__module__ == "satkit"
        assert len(set(members)) == len(members)
        assert {m: int(m) for m in members}[members[1]] == int(members[1])
        for m in members:
            assert m == int(m) and hash(m) == hash(int(m))
            assert pickle.loads(pickle.dumps(m)) == m
        assert sorted(int(m) for m in members) == list(range(len(members)))

    @pytest.mark.parametrize("cls", ENUMS)
    def test_frozen(self, cls):
        m = _members(cls)[0]
        with pytest.raises(AttributeError):
            m.value = 3

    def test_sgp4_error_from_code(self):
        for m in _members(sk.sgp4_error):
            assert sk.sgp4_error(int(m)) == m
        assert sk.sgp4_error(np.int32(6)) == sk.sgp4_error.orbit_decay
        for bad in (-1, 8):
            with pytest.raises(ValueError, match="sgp4_error code"):
                sk.sgp4_error(bad)
        with pytest.raises(TypeError):
            sk.sgp4_error("eccen")

    def test_error_array_maps_to_members(self):
        tle = _iss()
        _, _, err = sk.sgp4(tle, [tle.epoch, tle.epoch + sk.duration(days=1)], errflag=True)
        assert [sk.sgp4_error(int(e)) for e in err] == [sk.sgp4_error.success] * 2

    def test_fit_status_in_a_set(self):
        converged = {
            sk.tlefitstatus.GradientConverged,
            sk.tlefitstatus.StepConverged,
            sk.tlefitstatus.CostConverged,
        }
        for m in _members(sk.tlefitstatus):
            assert (m in converged) == m.converged


class TestReprAndPaths:
    """Review 4.8 and 4.6."""

    def test_repr_is_one_line(self):
        tle = _iss(named=True)
        assert repr(tle) == (
            'TLE(name="ISS (ZARYA)", satnum=25544, intl_desig="98067A", '
            "epoch=2021-09-21T12:25:40.104192Z)"
        )
        assert "\n" not in repr(sk.TLE())
        # str() keeps the multi-line listing
        assert "Inclination: 51.6442 deg" in str(tle)
        assert str(tle).count("\n") > 10

    def test_from_file_accepts_pathlike(self, tmp_path):
        path = tmp_path / "iss.tle"
        path.write_text("\n".join([ISS_NAME, *ISS]) + "\n")
        assert sk.TLE.from_file(path) == sk.TLE.from_file(str(path)) == [_iss(named=True)]
        with pytest.raises(TypeError):
            sk.TLE.from_file(3)  # type: ignore[arg-type]

    def test_omm_from_file_accepts_pathlike(self, tmp_path):
        import json

        path = tmp_path / "iss.json"
        path.write_text(json.dumps([_iss(named=True).to_omm()]))
        omms = sk.omm_from_file(path)
        assert omms == sk.omm_from_file(str(path))
        assert sk.TLE.from_omm(omms[0]) == _iss(named=True)
        with pytest.raises(TypeError):
            sk.omm_from_file(3)  # type: ignore[arg-type]


class TestFitResult:
    """Review 2.6 and 2.12: the result dict and the accepted time forms."""

    def test_keys_match_the_typeddict(self):
        from satkit._types import TLEFitResult

        tle = _iss()
        times = [tle.epoch + sk.duration(minutes=10 * i) for i in range(12)]
        pos, vel = sk.sgp4(tle, times)
        q = sk.frametransform.rotation(sk.frame.TEME, sk.frame.GCRF, times)
        states = [np.hstack((qi * p, qi * v)) for qi, p, v in zip(q, pos, vel)]
        fitted, result = sk.TLE.fit_from_states(states, times, tle.epoch)
        assert set(result) == set(TLEFitResult.__annotations__)
        assert isinstance(result["status"], sk.tlefitstatus)
        assert result["converged"] is result["status"].converged
        assert isinstance(result["n_iter"], int) and isinstance(result["best_norm"], float)
        assert (fitted.name, fitted.satnum, fitted.intl_desig) == ("none", 0, "")
        # times as a numpy datetime64 array
        times64 = np.array([t.to_datetime().replace(tzinfo=None) for t in times], "datetime64[us]")
        fitted64, _ = sk.TLE.fit_from_states(np.array(states), times64, tle.epoch)
        assert abs(fitted64.mean_motion - fitted.mean_motion) < 1e-9
