"""
Tests for the ECOM empirical solar-radiation-pressure model.

Unit-level checks (pickling, cannonball equivalence) plus a short orbit fit
against one day of real GPS SP3 truth that mirrors the Rust ``test_gps``
regression: reduced-ECOM must fit the observed orbit better than the
cannonball model.
"""

import os
import pickle
import struct

import numpy as np
import pytest
import satkit as sk

from sp3file import read_sp3file


def _gps_settings():
    s = sk.propsettings()
    s.gravity_degree = 12
    s.gravity_order = 12
    s.use_sun_gravity = True
    s.use_moon_gravity = True
    s.use_relativistic_correction = True
    s.use_spaceweather = False
    s.abs_error = 1e-11
    s.rel_error = 1e-11
    s.enable_interp = True
    return s


class TestEcomParams:
    def test_constructors_and_fields(self):
        e = sk.ecomparams.reduced(-1e-7, 1e-9, 2e-9, 3e-9, 4e-9)
        assert e.d0 == -1e-7 and e.bc == 3e-9 and e.bs == 4e-9
        assert e.sun_relative is False
        e2 = sk.ecomparams.ecom2(-1e-7, 0, 0, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9)
        assert e2.sun_relative is True
        assert (e2.bc, e2.bs, e2.d2c, e2.d4s) == (1e-9, 2e-9, 3e-9, 6e-9)
        kw = sk.ecomparams(d0=-2e-7, ys=7e-9, sun_relative=True)
        assert kw.d0 == -2e-7 and kw.ys == 7e-9 and kw.sun_relative
        kw.d0 = -3e-7
        assert kw.d0 == -3e-7
        assert sk.ecomparams() == sk.ecomparams()
        assert e != e2

    def test_dict_roundtrip(self):
        e = sk.ecomparams.ecom1(-1e-7, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9)
        d = e.to_dict()
        assert d["dc"] == 3e-9 and d["sun_relative"] is False
        assert sk.ecomparams.from_dict(d) == e
        assert sk.ecomparams.from_dict({"d0": -1e-7}).d0 == -1e-7

    def test_pickle_roundtrip(self):
        e = sk.ecomparams.ecom2(-1.1e-7, 1e-9, -2e-9, 3e-9, 4e-9, -5e-9, 6e-9, 7e-9, -8e-9)
        r = pickle.loads(pickle.dumps(e))
        assert r == e
        assert r.sun_relative is True

    def test_repr(self):
        e = sk.ecomparams.reduced(-1e-7, 0, 0, 0, 0)
        assert "d0=-1.0000e-7" in repr(e)
        assert "sun_relative=False" in repr(e)


class TestSatPropertiesEcom:
    def test_kwarg_and_property(self):
        e = sk.ecomparams.reduced(-1e-7, 1e-9, 2e-9, 3e-9, 4e-9)
        p = sk.satproperties(craoverm=0.0, ecom=e)
        assert p.ecom == e
        assert sk.satproperties().ecom is None
        p.ecom = None
        assert p.ecom is None
        p.ecom = e
        assert p.ecom.bc == 3e-9
        assert "ECOM" in str(p)

    def test_pickle_with_ecom(self):
        e = sk.ecomparams.ecom2(-1e-7, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9)
        t0 = sk.time(2024, 1, 1)
        t1 = t0 + sk.duration.from_hours(1)
        thr = sk.thrust.constant([1e-4, 0, 0], t0, t1, frame=sk.frame.RTN)
        p = sk.satproperties(craoverm=0.02, cdaoverm=0.01, thrusts=[thr], ecom=e)
        r = pickle.loads(pickle.dumps(p))
        assert r.craoverm == pytest.approx(0.02)
        assert r.cdaoverm == pytest.approx(0.01)
        assert len(r.thrusts) == 1 and r.thrusts[0].frame == sk.frame.RTN
        assert r.ecom == e
        # Without ECOM the v2 format still round-trips to None.
        r2 = pickle.loads(pickle.dumps(sk.satproperties(craoverm=0.02)))
        assert r2.ecom is None and r2.craoverm == pytest.approx(0.02)

    def test_v1_pickle_still_loads(self):
        """A satproperties pickled by a pre-ECOM release (format v1) must load."""
        v1 = struct.pack("<Bddi", 1, 0.02, 0.01, 0)  # version, craoverm, cdaoverm, 0 thrusts
        p = sk.satproperties()
        p.__setstate__(v1)
        assert p.craoverm == pytest.approx(0.02)
        assert p.cdaoverm == pytest.approx(0.01)
        assert p.thrusts == []
        assert p.ecom is None


class TestEcomPropagation:
    def test_d0_only_matches_cannonball(self):
        """d0 = -P_sun * Cr A/m with craoverm = 0 reproduces the cannonball model."""
        cr_a_over_m = 0.02
        t0 = sk.time(2024, 1, 15)
        t1 = t0 + sk.duration.from_days(1)
        settings = sk.propsettings()
        settings.abs_error = 1e-12
        settings.rel_error = 1e-12
        settings.use_spaceweather = False
        state = np.array([1.5e7, 2.0e7, 1.0e7, -2.5e3, 1.5e3, 1.0e3])
        cannon = sk.satproperties(craoverm=cr_a_over_m)
        ecom = sk.satproperties(craoverm=0.0, ecom=sk.ecomparams(d0=-4.56e-6 * cr_a_over_m))
        a = sk.propagate(state, t0, t1, propsettings=settings, satproperties=cannon).state
        b = sk.propagate(state, t0, t1, propsettings=settings, satproperties=ecom).state
        c = sk.propagate(state, t0, t1, propsettings=settings).state
        assert np.linalg.norm(a[:3] - b[:3]) < 1e-3
        assert np.linalg.norm(a[:3] - c[:3]) > 10.0

    def test_y_bias_is_cross_track(self):
        """A pure Y0 term should push mostly along e_Y = unit(e_D × r̂), never radially."""
        t0 = sk.time(2024, 1, 15)
        t1 = t0 + sk.duration.from_hours(3)
        settings = sk.propsettings()
        settings.use_spaceweather = False
        state = np.array([2.66e7, 0.0, 0.0, 0.0, 2.7e3, 2.7e3])
        base = sk.propagate(state, t0, t1, propsettings=settings).state
        ybias = sk.propagate(
            state, t0, t1, propsettings=settings,
            satproperties=sk.satproperties(ecom=sk.ecomparams(y0=5e-8)),
        ).state
        d = ybias[:3] - base[:3]
        assert np.linalg.norm(d) > 1.0


def test_reduced_ecom_fits_gps_better_than_cannonball(testvec_dir):
    """Fit state + SRP model to one day of ESA final SP3 (GPS PRN 20).

    Mirrors the Rust ``test_gps``: the reduced 5-parameter ECOM must fit
    the observed orbit better than a single cannonball coefficient.
    """
    least_squares = pytest.importorskip("scipy.optimize").least_squares
    approx_derivative = pytest.importorskip("scipy.optimize._numdiff").approx_derivative

    fname = os.path.join(testvec_dir, "orbitprop", "ESA0OPSFIN_20233640000_01D_05M_ORB.SP3")
    if not os.path.isfile(fname):
        pytest.skip(f"test vector {fname} not found")

    pitrf, times = read_sp3file(fname, satnum=20)
    times = list(times)
    truth = np.array([sk.frametransform.qitrf2gcrf(t) * p for t, p in zip(times, pitrf)])
    settings = _gps_settings()
    t0, t1 = times[0], times[-1]

    # Initial guess: first position, O(dt^4) one-sided stencil velocity
    # (a two-point chord is ~250 m/s off over a 5-min step at GPS altitude).
    dt = (times[1] - times[0]).seconds
    p = truth
    v0 = (-25 * p[0] + 48 * p[1] - 36 * p[2] + 16 * p[3] - 3 * p[4]) / (12 * dt)
    x0_state = np.concatenate((truth[0] / 1e3, v0))  # km, m/s

    def run(state_km_ms, props):
        st = np.concatenate((state_km_ms[:3] * 1e3, state_km_ms[3:]))
        res = sk.propagate(st, t0, t1, propsettings=settings, satproperties=props)
        return np.array(res.interp(times))[:, :3]

    def resid_cannon(x):
        return (run(x[:6], sk.satproperties(craoverm=x[6])) - truth).ravel()

    def resid_ecom(x):
        e = sk.ecomparams.reduced(*(x[6:11] * 1e-9))
        return (run(x[:6], sk.satproperties(craoverm=0.0, ecom=e)) - truth).ravel()

    # Absolute finite-difference steps (1 m, 1e-4 m/s, 0.1 nm/s^2 or
    # 1e-3 m^2/kg): scipy's `diff_step` is *relative* and silently falls
    # back to sqrt(eps) for zero-valued parameters, which is below
    # integrator noise for the harmonic ECOM coefficients.
    def jac_for(resid, coef_steps):
        steps = np.array([1e-3] * 3 + [1e-4] * 3 + coef_steps)
        return lambda x: approx_derivative(resid, x, abs_step=steps)

    kw = dict(x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=200)
    fit_c = least_squares(resid_cannon, np.concatenate((x0_state, [0.02])),
                          jac=jac_for(resid_cannon, [1e-3]), **kw)
    fit_e = least_squares(resid_ecom, np.concatenate((x0_state, [-100.0, 0, 0, 0, 0])),
                          jac=jac_for(resid_ecom, [0.1] * 5), **kw)
    rms_c = np.sqrt(np.mean(fit_c.fun**2))
    rms_e = np.sqrt(np.mean(fit_e.fun**2))
    print(f"cannonball fit RMS {rms_c:.3f} m (Cr A/m {fit_c.x[6]:.4f}); reduced-ECOM fit RMS {rms_e:.3f} m, "
          f"D0 {fit_e.x[6]:.1f} Y0 {fit_e.x[7]:.2f} B0 {fit_e.x[8]:.2f} Bc {fit_e.x[9]:.2f} Bs {fit_e.x[10]:.2f} nm/s^2")
    assert rms_e < rms_c
    assert rms_e < 1.0
    # D0 must be negative and of the expected magnitude (~ -P_sun * Cr A/m).
    assert -200.0 < fit_e.x[6] < -30.0
