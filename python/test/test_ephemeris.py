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
            time = sk.time.from_jd(jd, sk.timescale.TT)
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
        t1 = sk.time.from_mjd(t0.as_mjd(sk.timescale.UTC), sk.timescale.TDB)
        p = sk.moon.pos_gcrf(t1)
        ref_pos = np.array([-134240.626e3, -311571.590e3, -126693.785e3])
        assert p == pytest.approx(ref_pos)

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


class TestSun:
    def test_sunpos_mod(self):
        """
        Vallado example 5-1 for computing position of sun
        """
        t0 = sk.time(2006, 4, 2)
        # Vallado approximates UTC as TBD, so we will
        # make the same approximation
        # for the purposes of this test case
        t1 = sk.time.from_mjd(t0.as_mjd(sk.timescale.UTC), sk.timescale.TDB)
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
        (year, mon, day, hour, minute, sec) = sunrise.as_gregorian()
        assert year == 1996
        assert mon == 3
        assert day == 23
        assert hour == 5
        assert minute == 58
        assert sec == pytest.approx(21.97, 1e-3)
        (year, mon, day, hour, minute, sec) = sunset.as_gregorian()
        assert year == 1996
        assert mon == 3
        assert day == 23
        assert hour == 18
        assert minute == 15
        assert sec == pytest.approx(17.76, 1.0e-3)

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
