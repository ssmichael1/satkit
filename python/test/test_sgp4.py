import pytest
import numpy as np
import math as m
import os
import json
import pickle

import satkit as sk


class TestTLE:
    def test_tle_setting(self):
        """
        Test setting TLE parameters
        """
        line1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003"
        line2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
        tle = sk.TLE.from_lines([line1, line2])
        if isinstance(tle, list):
            tle = tle[0]
        assert tle.inclination == pytest.approx(51.6432, rel=1e-7)
        assert tle.raan == pytest.approx(351.4697, rel=1e-7)
        assert tle.eccen == pytest.approx(0.0007417, rel=1e-7)
        assert tle.arg_of_perigee == pytest.approx(130.5364, rel=1e-7)
        assert tle.mean_anomaly == pytest.approx(329.6482, rel=1e-7)
        assert tle.mean_motion == pytest.approx(15.48915330, rel=1e-7)
        assert tle.bstar == pytest.approx(0.00010270, rel=1e-4)
        assert abs((tle.epoch - sk.time(2021, 10, 2, 14, 10, 59.0)).seconds) < 1

        tle.raan = 50.0
        assert tle.raan == pytest.approx(50.0, rel=1e-7)
        tle.eccen = 0.1
        assert tle.eccen == pytest.approx(0.1, rel=1e-7)
        tle.arg_of_perigee = 40.0
        assert tle.arg_of_perigee == pytest.approx(40.0, rel=1e-7)
        tle.mean_anomaly = 300.0
        assert tle.mean_anomaly == pytest.approx(300.0, rel=1e-7)
        tle.mean_motion = 14.0
        assert tle.mean_motion == pytest.approx(14.0, rel=1e-7)
        tle.bstar = 0.0002
        assert tle.bstar == pytest.approx(0.0002, rel=1e-4)


    def test_tle_pickle(self):
        """TLE pickle must round-trip every serialized field."""
        line0 = "0 ISS (ZARYA)"
        line1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003"
        line2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
        tle = sk.TLE.from_lines([line0, line1, line2])
        if isinstance(tle, list):
            tle = tle[0]

        restored = pickle.loads(pickle.dumps(tle))

        assert restored.name == tle.name
        assert restored.satnum == tle.satnum
        assert restored.inclination == pytest.approx(tle.inclination, rel=1e-12)
        assert restored.raan == pytest.approx(tle.raan, rel=1e-12)
        assert restored.eccen == pytest.approx(tle.eccen, rel=1e-12)
        assert restored.arg_of_perigee == pytest.approx(tle.arg_of_perigee, rel=1e-12)
        assert restored.mean_anomaly == pytest.approx(tle.mean_anomaly, rel=1e-12)
        assert restored.mean_motion == pytest.approx(tle.mean_motion, rel=1e-12)
        assert restored.bstar == pytest.approx(tle.bstar, rel=1e-12)
        assert abs((restored.epoch - tle.epoch).seconds) < 1e-6
        # The restored TLE must propagate identically
        t = tle.epoch + sk.duration.from_hours(1.0)
        p1, v1 = sk.sgp4(tle, t)
        p2, v2 = sk.sgp4(restored, t)
        assert np.allclose(p1, p2) and np.allclose(v1, v2)

    def test_tle_pickle_rejects_garbage(self):
        """Malformed / pre-versioned pickle bytes raise ValueError, not a panic."""
        tle = sk.TLE.__new__(sk.TLE)
        with pytest.raises(ValueError):
            tle.__setstate__(b"aaaa")
        # A wrong version byte gets the explicit unsupported-version message
        with pytest.raises(ValueError, match="version"):
            tle.__setstate__(bytes([7] * 200))


class TestTLEFitting:
    def test_tle_fit(self):
        """
        Test fitting of TLE From high-precision states
        """
        altitude = 400e3
        r0 = sk.consts.earth_radius + altitude
        v0 = m.sqrt(sk.consts.mu_earth / r0)
        inc = 97 * m.pi / 180
        state0 = np.array([r0, 0, 0, 0, v0 * m.cos(inc), v0 * m.sin(inc)])

        sp = sk.satproperties(cdaoverm=2.0 * 10 / 3500)
        tm = sk.time(2016, 5, 16, 12, 0, 0)
        res = sk.propagate(
            state0, tm, end=tm + sk.duration.from_days(1), satproperties=sp
        )
        time_arr = [tm + sk.duration(seconds=x * 10) for x in range(8640)]
        state_arr = [res.interp(t) for t in time_arr]
        epoch = time_arr[0]

        _tle, _result = sk.TLE.fit_from_states(state_arr, time_arr, epoch)  # type: ignore


class TestSGP4:
    def test_sgp4_multiple(self):
        """
        Check propagating multiple TLEs at once
        """

        lines = [
            "0 STARLINK-3118",
            "1 49140U 21082L   24030.39663557  .00000076  00000-0  14180-4 0  9995",
            "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
            "0 STARLINK-3093",
            "1 49141U 21082M   24030.50141584 -.00000431  00000-0 -28322-4 0  9990",
            "2 49141  70.0000  73.8654 0002647 256.8611 103.2253 14.98324813131968",
            "0 STARLINK-3042",
            "1 49142U 21082N   24030.19218442  .00000448  00000-0  45331-4 0  9999",
            "2 49142  70.0005  34.6319 0002749 265.6056  94.4790 14.98327526131704",
            "0 STARLINK-3109",
            "1 49143U 21082P   24030.20076173 -.00000320  00000-0 -19071-4 0  9998",
            "2 49143  70.0002  54.6139 0002526 255.5608 104.5271 14.98327699131201",
        ]
        tles = sk.TLE.from_lines(lines)
        print(tles)
        tm = [
            sk.time(2024, 1, 15) + sk.duration.from_seconds(x * 10) for x in range(100)
        ]
        [p, v] = sk.sgp4(tles, tm)
        [p2, v2] = sk.sgp4(tles[2], tm)  # type: ignore
        # Verify that propagating multiple TLEs matches propagation of a single TLE
        assert p2 == pytest.approx(np.squeeze(p[2, :, :]))
        assert v2 == pytest.approx(np.squeeze(v[2, :, :]))

    def test_to_lines(self):
        """
        Test converting TLE to lines
        """

        lines = [
            "STARLINK-3118",
            "1 49140U 21082L   24030.39663557  .00000076  00000-0  14180-4 0  9995",
            "2 49140  70.0008  34.1139 0002663 260.3521  99.7337 14.98327656131736",
        ]
        tle = sk.TLE.from_lines(lines)

        if isinstance(tle, list):
            tle = tle[0]

        lines2 = tle.to_2line()
        assert lines[1:] == lines2

        lines2 = tle.to_3line()
        assert lines == lines2

    def test_omm(self, testvec_dir):
        """
        Test propagation of Orbital Mean-Element Message (OMM)
        which is represented as a dictionary
        """

        basedir = testvec_dir + os.path.sep + "omm" + os.path.sep

        # Plain json.load output is accepted as-is
        with open(basedir + "spacetrack_omm.json", "r") as fh:
            omm_list = json.load(fh)
        epoch = sk.time(omm_list[0]["EPOCH"])
        _p, _v = sk.sgp4(omm_list[0], epoch)
        p_list, _v = sk.sgp4(omm_list[0:3], epoch)
        assert p_list.shape == (3, 3)

        with open(basedir + "celestrak_omm.json", "r") as fh:
            omm_list = json.load(fh)
        epoch = sk.time(omm_list[0]["EPOCH"])
        _p, _v = sk.sgp4(omm_list[0], epoch)
        _p, _v = sk.sgp4(omm_list[0:3], epoch)

    def test_omm_loaders(self, testvec_dir):
        """
        omm_from_file / omm_from_text: JSON and XML, Space-Track and CelesTrak,
        must give the same dictionaries, with every field and extra kept
        """
        basedir = testvec_dir + os.path.sep + "omm" + os.path.sep

        st_json = sk.omm_from_file(basedir + "spacetrack_omm.json")
        st_xml = sk.omm_from_file(basedir + "spacetrack_omm.xml")
        assert len(st_json) == len(st_xml) == 2000
        j, x = st_json[0], st_xml[0]
        # Space-Track quotes every number; the loader returns numbers
        assert isinstance(j["MEAN_MOTION"], float)
        assert isinstance(j["NORAD_CAT_ID"], int)
        assert isinstance(x["MEAN_MOTION"], float)
        # Metadata, comments and user-defined extras all survive both formats
        for key in ("CCSDS_OMM_VERS", "COMMENT", "ORIGINATOR", "CREATION_DATE",
                    "TIME_SYSTEM", "MEAN_ELEMENT_THEORY", "OBJECT_TYPE",
                    "SEMIMAJOR_AXIS", "RCS_SIZE", "GP_ID"):
            assert j[key] == x[key], key
        assert j["RCS_SIZE"] is None
        # The JSON endpoint also carries the TLE lines; XML does not
        assert set(j) - set(x) == {"TLE_LINE0", "TLE_LINE1", "TLE_LINE2"}
        assert set(x) <= set(j)
        for key in x:
            assert x[key] == j[key], key

        ct_json = sk.omm_from_file(basedir + "celestrak_omm.json")
        ct_xml = sk.omm_from_file(basedir + "celestrak_omm.xml")
        assert len(ct_json) == len(ct_xml) == 149
        assert ct_json[0]["OBJECT_NAME"] == ct_xml[0]["OBJECT_NAME"]
        assert ct_json[0]["MEAN_MOTION"] == ct_xml[0]["MEAN_MOTION"]

        # Loaded dicts propagate; a JSON re-serialization round-trips
        epoch = sk.time(j["EPOCH"])
        p1, _v = sk.sgp4(j, epoch)
        p2, _v = sk.sgp4(x, epoch)
        p3, _v = sk.sgp4(sk.omm_from_text(json.dumps(j))[0], epoch)
        assert np.array_equal(p1, p2)
        assert np.array_equal(p1, p3)

        # A bare JSON object (not an array) is accepted too
        assert len(sk.omm_from_text(json.dumps(j))) == 1
        # KVN is not
        with pytest.raises(RuntimeError, match="KVN"):
            sk.omm_from_text("CCSDS_OMM_VERS = 3.0")

    def test_omm_matches_tle_lines(self, testvec_dir):
        """
        Space-Track ships the TLE lines with each OMM: both parsers must
        give the same SGP4 state (to the TLE's 0.864 ms epoch resolution)
        """
        basedir = testvec_dir + os.path.sep + "omm" + os.path.sep
        omms = sk.omm_from_file(basedir + "spacetrack_omm.json")
        checked = 0
        for rec in omms[:100]:
            tle = sk.TLE.from_lines([rec["TLE_LINE1"], rec["TLE_LINE2"]])
            tm = sk.time(rec["EPOCH"]) + sk.duration(hours=3)
            p_tle, _v, e_tle = sk.sgp4(tle, tm, errflag=True)
            p_omm, _v, e_omm = sk.sgp4(rec, tm, errflag=True)
            if e_tle[0] != 0 or e_omm[0] != 0:
                continue
            assert np.linalg.norm(p_tle - p_omm) < 10.0, rec["OBJECT_NAME"]
            checked += 1
        assert checked > 50

    def test_omm_dict_shapes(self):
        """
        sgp4 accepts flat CCSDS dicts and the nested groups of an
        xmltodict-style tree, with numbers as strings, and rejects what
        SGP4 cannot propagate
        """
        flat = {
            "OBJECT_NAME": "ISS (ZARYA)", "OBJECT_ID": "1998-067A",
            "EPOCH": "2021-10-02T14:10:59.999", "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
            "MEAN_MOTION": "15.48915330", "ECCENTRICITY": "0.0007417",
            "INCLINATION": "51.6432", "RA_OF_ASC_NODE": "351.4697",
            "ARG_OF_PERICENTER": "130.5364", "MEAN_ANOMALY": "329.6482",
            "BSTAR": "0.0001027", "MEAN_MOTION_DOT": "0.00016717",
            "MEAN_MOTION_DDOT": "0",
        }
        epoch = sk.time(flat["EPOCH"])
        tm = epoch + sk.duration(minutes=30)
        p_ref, _v = sk.sgp4(flat, tm)

        # Nested (xmltodict) form, handed in at the omm, body, or data level
        nested = {
            "@id": "CCSDS_OMM_VERS", "@version": "2.0",
            "body": {"segment": {
                "metadata": {k: flat[k] for k in ("OBJECT_NAME", "OBJECT_ID",
                                                  "TIME_SYSTEM", "MEAN_ELEMENT_THEORY")},
                "data": {
                    "meanElements": {k: flat[k] for k in (
                        "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
                        "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY")},
                    "tleParameters": {k: flat[k] for k in (
                        "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT")},
                    "userDefinedParameters": {"USER_DEFINED": [
                        {"@parameter": "OBJECT_TYPE", "#text": "PAYLOAD"},
                        {"@parameter": "RCS_SIZE"},
                    ]},
                },
            }},
        }
        for node in (nested, nested["body"], nested["body"]["segment"]["data"]):
            p, _v = sk.sgp4(node, tm)
            assert np.array_equal(p, p_ref)

        # A trimmed dict with only the elements, epoch as satkit.time / datetime
        minimal = {k: float(flat[k]) for k in (
            "MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
            "ARG_OF_PERICENTER", "MEAN_ANOMALY")}
        minimal["BSTAR"] = float(flat["BSTAR"])
        minimal["MEAN_MOTION_DOT"] = float(flat["MEAN_MOTION_DOT"])
        for ep in (epoch, epoch.as_datetime(), flat["EPOCH"]):
            p, _v = sk.sgp4({**minimal, "EPOCH": ep}, tm)
            assert np.linalg.norm(p - p_ref) < 1e-6

        # Optional fields may be None or empty; metadata is case-insensitive
        p, _v = sk.sgp4({**flat, "BSTAR": None, "MEAN_MOTION_DDOT": "",
                         "MEAN_ELEMENT_THEORY": " sgp4 "}, tm)
        assert np.linalg.norm(p - p_ref) < 1e3  # BSTAR gone: small drag change

        with pytest.raises(RuntimeError, match="EPOCH"):
            sk.sgp4({}, tm)
        with pytest.raises(RuntimeError, match="MEAN_MOTION"):
            sk.sgp4({**flat, "MEAN_MOTION": "fast"}, tm)
        with pytest.raises(RuntimeError, match="EPHEMERIS_TYPE 4"):
            sk.sgp4({**flat, "EPHEMERIS_TYPE": 4}, tm)
        with pytest.raises(RuntimeError, match="MEAN_ELEMENT_THEORY"):
            sk.sgp4({**flat, "MEAN_ELEMENT_THEORY": "DSST"}, tm)
        with pytest.raises(RuntimeError, match="TIME_SYSTEM"):
            sk.sgp4({**flat, "TIME_SYSTEM": "TAI"}, tm)

    def test_tle_omm_conversion(self):
        """
        TLE.to_omm / TLE.from_omm round-trip and agree with sgp4 on both
        """
        line0 = "0 ISS (ZARYA)"
        line1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003"
        line2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
        tle = sk.TLE.from_lines([line0, line1, line2])

        omm = tle.to_omm()
        assert omm["OBJECT_NAME"] == "ISS (ZARYA)"
        assert omm["OBJECT_ID"] == "1998-067A"
        assert omm["NORAD_CAT_ID"] == 25544
        assert omm["ELEMENT_SET_NO"] == 900
        assert omm["REV_AT_EPOCH"] == 29935
        assert omm["EPHEMERIS_TYPE"] == 0
        assert omm["MEAN_ELEMENT_THEORY"] == "SGP4"
        assert omm["MEAN_MOTION"] == tle.mean_motion
        assert omm["BSTAR"] == tle.bstar
        assert sk.time(omm["EPOCH"]) == tle.epoch
        assert "CLASSIFICATION_TYPE" not in omm
        json.dumps(omm)  # plain JSON-serializable values only

        back = sk.TLE.from_omm(omm)
        assert back.to_2line() == tle.to_2line()
        assert back.intl_desig == "98067A"
        assert back.name == "ISS (ZARYA)"

        tm = tle.epoch + sk.duration(hours=1)
        p_tle, v_tle = sk.sgp4(tle, tm)
        p_omm, v_omm = sk.sgp4(omm, tm)
        p_back, v_back = sk.sgp4(back, tm)
        assert np.array_equal(p_tle, p_omm)
        assert np.array_equal(p_tle, p_back)

        # Pickling the converted TLE keeps the designator
        assert pickle.loads(pickle.dumps(back)).intl_desig == "98067A"

    def test_sgp4_vallado(self, testvec_dir):
        """
        SGP4 Test Vectors from vallado
        """

        basedir = testvec_dir + os.path.sep + "sgp4"

        tlefile = basedir + os.path.sep + "SGP4-VER.TLE"
        with open(tlefile, "r") as fh:
            lines = fh.readlines()

        lines = list(filter(lambda x: x[0] != "#", lines))

        lines = [l.strip() for l in lines]
        lines = [l[0:69] for l in lines]

        tles = sk.TLE.from_lines(lines)
        for tle in tles:  # type: ignore
            fname = f"{basedir}{os.path.sep}{tle.satnum:05}.e"
            with open(fname, "r") as fh:
                testvecs = fh.readlines()
            for testvec in testvecs:
                stringvals = testvec.split()

                # Valid lines are all floats of length 7
                if len(stringvals) != 7:
                    continue
                try:
                    vals = [float(s) for s in stringvals]
                except ValueError:
                    continue
                time = tle.epoch + sk.duration.from_seconds(vals[0])
                try:
                    [p, v, eflag] = sk.sgp4(  # type: ignore
                        tle,
                        time,
                        opsmode=sk.sgp4_opsmode.afspc,
                        gravconst=sk.sgp4_gravconst.wgs72,
                        errflag=True,
                    )

                    if eflag == sk.sgp4_error.success:
                        ptest = np.array([vals[1], vals[2], vals[3]]) * 1e3
                        vtest = np.array([vals[4], vals[5], vals[6]]) * 1e3
                        assert p == pytest.approx(ptest, rel=1e-4)
                        assert v == pytest.approx(vtest, rel=1e-2)
                    else:
                        # We know which one is supposed to fail in the test vectors
                        # Make sure we pick the correcxt one
                        assert tle.satnum == 33334
                        assert eflag == sk.sgp4_error.perturb_eccen
                except RuntimeError:
                    print("Caught runtime error; this is expected in test vectors")


class TestTLEMetadata:
    def test_tle_metadata_getters(self):
        """The catalog-identity fields must be readable (and settable)."""
        line0 = "0 ISS (ZARYA)"
        line1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003"
        line2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
        tle = sk.TLE.from_lines([line0, line1, line2])
        if isinstance(tle, list):
            tle = tle[0]
        assert tle.intl_desig == "98067A"
        assert tle.desig_year == 98
        assert tle.desig_launch == 67
        assert tle.desig_piece == "A"
        assert tle.element_num == 900
        assert tle.rev_num == 29935
        assert tle.ephem_type == 0
        tle.element_num = 901
        assert tle.element_num == 901


class TestSGP4ListKwargs:
    def test_list_honors_gravconst(self):
        """The list path must honor gravconst/opsmode kwargs (previously
        silently ignored): list results must match per-TLE results computed
        with the same non-default settings."""
        line1 = "1 25544U 98067A   21275.59097222  .00016717  00000-0  10270-3 0  9003"
        line2 = "2 25544  51.6432 351.4697 0007417 130.5364 329.6482 15.48915330299357"
        def one(tles):
            return tles[0] if isinstance(tles, list) else tles

        tle_a = one(sk.TLE.from_lines([line1, line2]))
        tle_b = one(sk.TLE.from_lines([line1, line2]))
        t = tle_a.epoch + sk.duration.from_hours(6)

        p_single, v_single = sk.sgp4(tle_a, t, gravconst=sk.sgp4_gravconst.wgs84)
        p_list, v_list = sk.sgp4([tle_b], [t], gravconst=sk.sgp4_gravconst.wgs84)
        assert np.allclose(np.asarray(p_list).squeeze(), p_single)
        assert np.allclose(np.asarray(v_list).squeeze(), v_single)

        # And wgs84 must actually differ from the default wgs72
        p_72, _ = sk.sgp4(one(sk.TLE.from_lines([line1, line2])), t)
        assert not np.allclose(p_72, p_single, rtol=0, atol=1e-3)
