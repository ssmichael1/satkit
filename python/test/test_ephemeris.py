import pytest
import numpy as np

import satkit as sk


class TestJPLEphem:
    def test_jplephem_testvecs(self, testvec_dir):
        """
        Test JPL ephemeris against test vectors provided by JPL
        """

        import os

        # File contains test calculation vectors provided by NASA
        # for the JPL DE440 ephemeris

        fname = testvec_dir + os.path.sep + "jplephem" + os.path.sep + "testpo.440"

        # Read in the test vectors
        with open(fname, "r") as fd:
            lines = fd.readlines()

        # Function to convert integer index to solar system body
        def int_to_ss(ix: int) -> sk.solarsystem:
            if ix == 0:
                return sk.solarsystem.Mercury
            elif ix == 1:
                return sk.solarsystem.Venus
            elif ix == 2:
                return sk.solarsystem.EMB
            elif ix == 3:
                return sk.solarsystem.Mars
            elif ix == 4:
                return sk.solarsystem.Jupiter
            elif ix == 5:
                return sk.solarsystem.Saturn
            elif ix == 6:
                return sk.solarsystem.Uranus
            elif ix == 7:
                return sk.solarsystem.Neptune
            elif ix == 8:
                return sk.solarsystem.Pluto
            elif ix == 9:
                return sk.solarsystem.Moon
            elif ix == 10:
                return sk.solarsystem.Sun
            else:
                raise ValueError(f"Unknown solar system body index: {ix}")

        # Go through the test vectors
        # each test vecxtor is a line in the file
        for line in lines[14:]:
            s = line.split()
            assert len(s) >= 7
            # get the fields in the test vector
            jd = float(s[2])
            tar = int(s[3])
            src = int(s[4])
            coord = int(s[5])
            truth = float(s[6])
            # testpo epochs are JD in T_eph (TDB)
            time = sk.time.from_jd(jd, sk.timescale.TDB)
            # Don't handle any of the exotic test vectors, just do sun, moon,
            # and planetary ephemerides
            if tar <= 10 and src <= 10 and coord <= 6:
                sksrc = int_to_ss(src - 1)
                sktar = int_to_ss(tar - 1)
                tpos, tvel = sk.jplephem.geocentric_state(sktar, time)
                spos, svel = sk.jplephem.geocentric_state(sksrc, time)

                # In test vectors, index 3 is not EMB, but Earth
                # (not obvious...)
                if tar == 3:
                    _mpos, mvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.Moon, time
                    )
                    tvel = tvel - mvel / (1.0 + sk.consts.earth_moon_mass_ratio)
                    tpos = np.array([0, 0, 0])
                if src == 3:
                    spos = np.array([0, 0, 0])
                    _mpos, mvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.Moon, time
                    )
                    svel = svel - mvel / (1.0 + sk.consts.earth_moon_mass_ratio)
                if src == 10:
                    embpos, embvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.EMB, time
                    )
                    svel = svel + (
                        embvel - svel / (1.0 + sk.consts.earth_moon_mass_ratio)
                    )
                if tar == 10:
                    embpos, embvel = sk.jplephem.geocentric_state(
                        sk.solarsystem.EMB, time
                    )
                    tvel = tvel + (
                        embvel - tvel / (1.0 + sk.consts.earth_moon_mass_ratio)
                    )
                # Position test
                if coord <= 3:
                    calc = (tpos - spos)[coord - 1] / sk.consts.au
                    assert calc == pytest.approx(truth, rel=1e-12)
                # Velocity test
                else:
                    calc = (tvel - svel)[coord - 4] / sk.consts.au * 86400.0
                    assert calc == pytest.approx(truth, rel=1e-12)


class TestMoon:
    def test_moonpos(self):
        """
        Vallado example 5-3 for
        computing position of the moon
        """
        t0 = sk.time(1994, 4, 28)
        # Vallado approximates UTC as TBD, so we will
        # make the same approximation
        # for the purposes of this test case
        t1 = sk.time.from_mjd(t0.to_mjd(sk.timescale.UTC), sk.timescale.TDB)
        p = sk.moon.pos_gcrf(t1)
        # Vallado's worked example is in mean-of-date coordinates
        ref_pos = np.array([-134240.626e3, -311571.590e3, -126693.785e3])
        assert p == pytest.approx(sk.frametransform.qmod2gcrf(t1) * ref_pos)

    def test_moonpos_vs_jplephem(self):
        # pos_gcrf used to return mean-of-date coordinates, off from GCRF by
        # precession (~1.4 deg / century from J2000)
        t0 = sk.time(1950, 1, 1)
        times = [t0 + sk.duration(days=d) for d in np.arange(0.0, 150 * 365.25, 4.37)]
        lp = sk.moon.pos_gcrf(times)
        jpl = sk.jplephem.geocentric_pos(sk.solarsystem.Moon, times)
        # J2000 ecliptic longitude / latitude
        eps = np.radians(84381.406 / 3600.0)
        rot = np.array(
            [[1, 0, 0], [0, np.cos(eps), np.sin(eps)], [0, -np.sin(eps), np.cos(eps)]]
        )

        def lonlat(p):
            e = p @ rot.T
            return np.arctan2(e[:, 1], e[:, 0]), np.arcsin(e[:, 2] / np.linalg.norm(e, axis=1))

        (l1, b1), (l2, b2) = lonlat(lp), lonlat(jpl)
        dlon = np.degrees(np.angle(np.exp(1j * (l1 - l2))))
        assert np.max(np.abs(dlon)) < 0.37
        assert np.max(np.abs(np.degrees(b1 - b2))) < 0.2
        years = np.arange(len(times)) * 4.37 / 365.25
        assert abs(np.polyfit(years, dlon, 1)[0]) < 1.0e-4  # deg / year

    def test_moon_phase(self):
        # Checked against https://www.timeanddate.com/moon/phases/
        t = sk.time(2025, 11, 12, 0, 46, 0)
        phasename = sk.moon.phase_name(t)
        assert phasename == sk.moon.moonphase.LastQuarter
        illumination = sk.moon.illumination(t)
        assert illumination == pytest.approx(0.52, rel=1e-2)

    def test_moon_phase_name(self):
        """
        Test moon phase name function
        against known moon phases in early 2024

        phases compared against results from https://www.moongiant.com/
        """

        # New moon - Jan 11 2024
        t_new = sk.time(2024, 1, 11, 17, 8, 0)
        phase_new = sk.moon.phase_name(t_new)
        assert phase_new == sk.moon.moonphase.NewMoon

        # First quarter - Jan 18 2024
        t_first = sk.time(2024, 1, 18, 12, 0, 0)
        phase_first = sk.moon.phase_name(t_first)
        assert phase_first == sk.moon.moonphase.FirstQuarter

        # Full moon - Jan 25 2024
        t_full = sk.time(2024, 1, 25, 4, 54, 0)
        phase_full = sk.moon.phase_name(t_full)
        assert phase_full == sk.moon.moonphase.FullMoon

        # Last quarter - Feb 2 2024
        t_last = sk.time(2024, 2, 2, 2, 0, 0)
        phase_last = sk.moon.phase_name(t_last)
        assert phase_last == sk.moon.moonphase.LastQuarter


class TestPlanets:
    @pytest.mark.parametrize(
        "planet, start, lon_rms, lat_rms",
        [
            # JPL's approximate errors (arcsec) for the element set in use
            (sk.solarsystem.Mercury, 1800, 15, 1),
            (sk.solarsystem.Mars, 1800, 40, 2),
            # 2051-2650 uses the 3000 BC - 3000 AD elements plus the extra
            # mean-anomaly terms, which were applied in the wrong units
            (sk.solarsystem.Jupiter, 2051, 600, 100),
            (sk.solarsystem.Saturn, 2051, 1000, 100),
        ],
    )
    def test_heliocentric_pos_vs_jplephem(self, planet, start, lon_rms, lat_rms):
        t0 = sk.time(start, 1, 2)
        span = 249 if start == 1800 else 598
        times = [t0 + sk.duration(days=d) for d in np.arange(0.0, span * 365.25, 11.3)]
        lp = sk.planets.heliocentric_pos(planet, times)
        jpl = sk.jplephem.barycentric_pos(planet, times) - sk.jplephem.barycentric_pos(
            sk.solarsystem.Sun, times
        )
        # J2000 ecliptic, the frame of JPL's error table
        eps = np.radians(23.43928)
        rot = np.array(
            [[1, 0, 0], [0, np.cos(eps), np.sin(eps)], [0, -np.sin(eps), np.cos(eps)]]
        )
        e1, e2 = lp @ rot.T, jpl @ rot.T
        dlon = np.angle(np.exp(1j * (np.arctan2(e1[:, 1], e1[:, 0]) - np.arctan2(e2[:, 1], e2[:, 0]))))
        dlat = np.arcsin(e1[:, 2] / np.linalg.norm(e1, axis=1)) - np.arcsin(
            e2[:, 2] / np.linalg.norm(e2, axis=1)
        )
        assert np.sqrt(np.mean(np.degrees(dlon) ** 2)) * 3600 < lon_rms
        assert np.sqrt(np.mean(np.degrees(dlat) ** 2)) * 3600 < lat_rms


class TestSun:
    def test_sunpos_mod(self):
        """
        Vallado example 5-1 for computing position of sun
        """
        t0 = sk.time(2006, 4, 2)
        # Vallado approximates UTC as TBD, so we will
        # make the same approximation
        # for the purposes of this test case
        t1 = sk.time.from_mjd(t0.to_mjd(sk.timescale.UTC), sk.timescale.TDB)
        p = sk.sun.pos_gcrf(t1)
        pref = np.array([146259922.0e3, 28585947.0e3, 12397430.0e3])
        assert p == pytest.approx(pref, 5e-4)

    def test_sun_rise_set(self):
        """
        Vallado example 5-2
        """
        coord = sk.itrfcoord(latitude_deg=40.0, longitude_deg=0.0)
        tm = sk.time(1996, 3, 23, 0, 0, 0)
        sunrise, sunset = sk.sun.rise_set(tm, coord)
        (year, mon, day, hour, minute, sec) = sunrise.to_gregorian()
        assert year == 1996
        assert mon == 3
        assert day == 23
        assert hour == 5
        assert minute == 58
        assert sec == pytest.approx(21.97, 1e-3)
        (year, mon, day, hour, minute, sec) = sunset.to_gregorian()
        assert year == 1996
        assert mon == 3
        assert day == 23
        assert hour == 18
        assert minute == 15
        assert sec == pytest.approx(17.76, 1.0e-3)

    def test_sun_rise_set_utc_date(self):
        # Any time on 2024-10-14 UTC gives that date's events; the old day
        # selection returned the next day for about half the inputs
        greenwich = sk.itrfcoord(latitude_deg=51.48, longitude_deg=0.0)
        honolulu = sk.itrfcoord(latitude_deg=21.31, longitude_deg=-157.86)
        for coord, rise_utc, set_utc in [
            (greenwich, (14, 6), (14, 17)),
            (honolulu, (14, 16), (15, 4)),
        ]:
            ref = sk.sun.rise_set(sk.time(2024, 10, 14), coord)
            for hour in (0, 6, 12, 18, 23):
                rs = sk.sun.rise_set(sk.time(2024, 10, 14, hour, 59, 59), coord)
                assert rs == ref
            assert ref[0].to_gregorian()[2:4] == rise_utc
            assert ref[1].to_gregorian()[2:4] == set_utc

    def test_sun_rise_set_local_noon(self):
        # A timezone-aware datetime at local noon selects that local date
        import datetime

        for lat, lon, utc_offset in [(21.31, -157.86, -10), (35.68, 139.69, 9)]:
            tz = datetime.timezone(datetime.timedelta(hours=utc_offset))
            coord = sk.itrfcoord(latitude_deg=lat, longitude_deg=lon)
            noon = datetime.datetime(2024, 10, 14, 12, tzinfo=tz)
            rise, set = sk.sun.rise_set(noon, coord)
            rise, set = rise.to_datetime().astimezone(tz), set.to_datetime().astimezone(tz)
            assert rise.date() == set.date() == datetime.date(2024, 10, 14)
            assert 5 <= rise.hour <= 6 and 17 <= set.hour <= 18

    def test_shadowfunc_annular(self):
        # Beyond the umbra on the anti-Sun axis the eclipse is annular,
        # 1 - b^2/a^2; this used to return NaN
        au = sk.consts.au
        sun = np.array([au, 0.0, 0.0])
        for d in (1.5e9, 2.0e9):
            a = np.arcsin(sk.consts.sun_radius / (au + d))
            b = np.arcsin(sk.consts.earth_radius / d)
            f = sk.sun.shadowfunc(sun, np.array([-d, 0.0, 0.0]))
            assert f == pytest.approx(1.0 - (b / a) ** 2, rel=1e-12)
        # Inside the Earth: never NaN
        assert sk.sun.shadowfunc(sun, np.array([-6.0e6, 0.0, 0.0])) == 0.0
        assert sk.sun.shadowfunc(sun, np.array([6.0e6, 0.0, 0.0])) == 1.0

    def test_sun_rise_set_error(self):
        coord = sk.itrfcoord(latitude_deg=85.0, longitude_deg=30.0)
        tm = sk.time(2020, 6, 20)
        try:
            sunrise, sunset = sk.sun.rise_set(tm, coord)
        except:
            # This should throw exception ... there is no sunrise or sunset
            # at this time of year at the specified location; sun is always up
            pass
        else:
            assert 1 == 0
