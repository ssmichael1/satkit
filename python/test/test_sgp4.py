import pytest
import numpy as np
import math as m
import os
import json
import pickle
import xmltodict

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
        fname = basedir + "spacetrack_omm.json"
        with open(fname, "r") as fh:
            omm_list = json.load(fh)
        epoch = sk.time(omm_list[0]["EPOCH"])
        # Run SGP4 on first OMM in list
        _p, _v = sk.sgp4(omm_list[0], epoch)
        # Run SGP4 on list of OMMs
        _p, _v = sk.sgp4(omm_list[0:3], epoch)

        fname = basedir + "celestrak_omm.json"
        with open(fname, "r") as fh:
            omm_list = json.load(fh)
        epoch = sk.time(omm_list[0]["EPOCH"])
        # Run SGP4 on first OMM in list
        _p, _v = sk.sgp4(omm_list[0], epoch)
        # Run SPG4 on list of OMMs
        _p, _v = sk.sgp4(omm_list[0:3], epoch)

        # Now try XML files
        fname = basedir + "spacetrack_omm.xml"
        with open(fname, "r") as fh:
            omm_xml = xmltodict.parse(fh.read())
        omm_xml = omm_xml["ndm"]["omm"]
        omm_xml = [d["body"]['segment']['data'] for d in omm_xml]
        epoch = sk.time(omm_xml[0]["meanElements"]["EPOCH"])
        _p, _v = sk.sgp4(omm_xml[0], epoch)
        _p, _v = sk.sgp4(omm_xml[0:3], epoch)

        fname = basedir + "celestrak_omm.xml"
        with open(fname, "r") as fh:
            omm_xml = xmltodict.parse(fh.read())
        omm_xml = omm_xml["ndm"]["omm"]
        omm_xml = [d["body"]['segment']['data'] for d in omm_xml]
        epoch = sk.time(omm_xml[0]["meanElements"]["EPOCH"])
        _p, _v = sk.sgp4(omm_xml[0], epoch)
        _p, _v = sk.sgp4(omm_xml[0:3], epoch)


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


class TestSGP4XP:
    def test_type4_tle_rejected(self):
        """An SGP4-XP element set (ephemeris type 4) parses but cannot be propagated."""
        line1 = "1 00011U 59001A   23060.12028874 +.00002871  89876-2  73526-1 4 00010"
        line2 = "2 00011  32.8652 309.4507 1466152  63.9843 312.2337 11.85947359392148"
        tle = sk.TLE.from_lines([line1, line2])
        if isinstance(tle, list):
            tle = tle[0]
        assert tle.ephem_type == 4
        with pytest.raises(RuntimeError, match="SGP4-XP"):
            sk.sgp4(tle, tle.epoch)

    def test_type4_omm_rejected(self):
        """EPHEMERIS_TYPE 4 in an OMM is rejected the same way."""
        omm = {
            "OBJECT_NAME": "VANGUARD 1",
            "OBJECT_ID": "1958-002B",
            "EPOCH": "2023-03-01T02:53:12.947",
            "MEAN_MOTION": 10.84532081,
            "ECCENTRICITY": 0.1849434,
            "INCLINATION": 34.2437,
            "RA_OF_ASC_NODE": 69.7250,
            "ARG_OF_PERICENTER": 110.9844,
            "MEAN_ANOMALY": 271.4417,
            "EPHEMERIS_TYPE": 4,
            "NORAD_CAT_ID": 5,
        }
        with pytest.raises(RuntimeError, match="SGP4-XP"):
            sk.sgp4(omm, sk.time(2023, 3, 1, 3, 0, 0))
