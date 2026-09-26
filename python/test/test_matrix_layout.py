"""Matrix memory-layout tests for every binding that passes a matrix across
the Rust/Python boundary.

satkit's Rust matrices (numeris) are column-major; numpy arrays built by
reshaping a flat buffer are row-major. Handing one's raw buffer to the other
silently transposes the matrix, and for a rotation that means applying the
inverse. This has happened at least twice: ``frametransform.to_gcrf`` once
returned the transposed DCM, and ``quaternion * V`` for an Nx3 array applied
the inverse rotation from 0.14.1 to 0.23.1.

Each test compares a binding's matrix against a reference built independently
of that binding's own matrix code (a single-vector rotation, a geometric
construction, or a finite difference), using inputs whose matrices are not
symmetric, so a transposition cannot pass. Symmetric outputs (covariance, the
gravity gradient) are listed at the end: layout cannot be observed there.
"""

import numpy as np
import pytest

import satkit as sk
from shared import ISS_2024, STARLINK_3118

RNG = np.random.default_rng(20260925)


def _random_quaternions(n):
    out = []
    for _ in range(n):
        axis = RNG.normal(size=3)
        out.append(sk.quaternion.from_axis_angle(axis, RNG.uniform(0.1, 3.0)))
    return out


class TestQuaternion:
    def test_known_rotation_matrix(self):
        """rotz(90 deg) has the non-symmetric matrix [[0,-1,0],[1,0,0],[0,0,1]]."""
        R = sk.quaternion.rotz(np.pi / 2).to_rotation_matrix()
        np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)

    @pytest.mark.parametrize("q", _random_quaternions(8))
    def test_rotation_matrix_matches_vector_rotation(self, q):
        """R @ v equals q * v (the single-vector path) for every basis vector."""
        R = q.to_rotation_matrix()
        for v in np.eye(3):
            np.testing.assert_allclose(R @ v, q * v, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("q", _random_quaternions(8))
    def test_from_rotation_matrix(self, q):
        """from_rotation_matrix(R) rotates vectors the same way as R."""
        R = q.to_rotation_matrix()
        q2 = sk.quaternion.from_rotation_matrix(R)
        for v in np.eye(3):
            np.testing.assert_allclose(q2 * v, R @ v, rtol=0, atol=1e-14)
        # and a known non-symmetric matrix
        qz = sk.quaternion.from_rotation_matrix(np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]))
        np.testing.assert_allclose(qz * np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-15)


class TestSatelliteFrames:
    """to_gcrf / from_gcrf DCMs against the RTN axes built from r and v."""

    pos = np.array([6_778_137.0, 1_234_567.0, -987_654.0])
    vel = np.array([-1_234.5, 7_300.0, 1_100.0])

    def _rtn_axes(self):
        r_hat = self.pos / np.linalg.norm(self.pos)
        h = np.cross(self.pos, self.vel)
        n_hat = h / np.linalg.norm(h)
        t_hat = np.cross(n_hat, r_hat)
        return r_hat, t_hat, n_hat

    def test_to_gcrf_columns_are_rtn_axes(self):
        dcm = sk.frametransform.to_gcrf(sk.frame.RTN, self.pos, self.vel)
        r_hat, t_hat, n_hat = self._rtn_axes()
        np.testing.assert_allclose(dcm[:, 0], r_hat, atol=1e-14)
        np.testing.assert_allclose(dcm[:, 1], t_hat, atol=1e-14)
        np.testing.assert_allclose(dcm[:, 2], n_hat, atol=1e-14)

    def test_from_gcrf_maps_position_to_radial(self):
        dcm = sk.frametransform.from_gcrf(sk.frame.RTN, self.pos, self.vel)
        np.testing.assert_allclose(dcm @ self.pos, [np.linalg.norm(self.pos), 0, 0], atol=1e-6)
        np.testing.assert_allclose(
            dcm, sk.frametransform.to_gcrf(sk.frame.RTN, self.pos, self.vel).T, atol=1e-15
        )


class TestStateArrays:
    """Nx3 / Nx6 array paths against row-by-row scalar calls."""

    @pytest.mark.parametrize(
        "fn",
        [
            sk.frametransform.itrf_to_gcrf_state,
            sk.frametransform.gcrf_to_itrf_state,
            sk.frametransform.itrf_to_gcrf_state_approx,
            sk.frametransform.gcrf_to_itrf_state_approx,
        ],
        ids=["itrf_to_gcrf", "gcrf_to_itrf", "itrf_to_gcrf_approx", "gcrf_to_itrf_approx"],
    )
    def test_batch_state_transform_rows(self, fn):
        """Nx3 position/velocity with N times equals N single-state calls."""
        t0 = sk.time(2024, 6, 15, 12, 0, 0)
        times = [t0 + sk.duration(minutes=17 * i) for i in range(5)]
        P = RNG.normal(size=(5, 3)) * 7e6
        V = RNG.normal(size=(5, 3)) * 7e3
        p_all, v_all = fn(P, V, times)
        for i in range(5):
            p1, v1 = fn(P[i], V[i], times[i])
            np.testing.assert_allclose(p_all[i], p1, rtol=0, atol=1e-8)
            np.testing.assert_allclose(v_all[i], v1, rtol=0, atol=1e-11)

    def test_sgp4_tle_by_time_grid(self):
        tles = sk.TLE.from_lines(ISS_2024 + STARLINK_3118)
        times = [tles[0].epoch + sk.duration(minutes=m) for m in (0, 17, 93)]
        p, v = sk.sgp4(tles, times)
        assert p.shape[-1] == 3
        for i, tle in enumerate(tles):
            for j, tm in enumerate(times):
                p1, v1 = sk.sgp4(tle, tm)
                np.testing.assert_allclose(p[i, j], p1, rtol=0, atol=1e-6)
                np.testing.assert_allclose(v[i, j], v1, rtol=0, atol=1e-9)


class TestStateTransitionMatrix:
    """propresult.phi and interp(..., output_phi=True) against finite differences.

    Over 60 s, d(position)/d(v0) is about 60*I while d(velocity)/d(r0) is
    tiny, so a transposed 6x6 STM fails the column comparison by orders of
    magnitude.
    """

    t0 = sk.time(2024, 1, 1)
    state0 = np.array([6_778_137.0, 0.0, 0.0, 0.0, 7_668.6, 0.0])
    dt = sk.duration(seconds=60)

    def _fd_column(self, j, h):
        s_plus = self.state0.copy()
        s_plus[j] += h
        s_minus = self.state0.copy()
        s_minus[j] -= h
        end = self.t0 + self.dt
        rp = sk.propagate(s_plus, self.t0, end).state_end
        rm = sk.propagate(s_minus, self.t0, end).state_end
        return (rp - rm) / (2 * h)

    def _check(self, phi):
        for j, h in ((0, 10.0), (4, 0.01)):
            np.testing.assert_allclose(phi[:, j], self._fd_column(j, h), rtol=1e-5, atol=1e-7)

    def test_propresult_phi(self):
        r = sk.propagate(self.state0, self.t0, self.t0 + self.dt, output_phi=True)
        self._check(r.phi)

    def test_interp_phi(self):
        r = sk.propagate(
            self.state0,
            self.t0,
            self.t0 + sk.duration(seconds=120),
            output_phi=True,
            propsettings=sk.propsettings(enable_interp=True),
        )
        _, phi = r.interp(self.t0 + self.dt, output_phi=True)
        self._check(phi)

    def test_interp_many_times_matches_scalar(self):
        r = sk.propagate(
            self.state0,
            self.t0,
            self.t0 + sk.duration(seconds=600),
            propsettings=sk.propsettings(enable_interp=True),
        )
        times = [self.t0 + sk.duration(seconds=s) for s in (30, 200, 450)]
        many = np.asarray(r.interp(times))
        assert many.shape == (3, 6)
        for i, tm in enumerate(times):
            np.testing.assert_allclose(many[i], r.interp(tm), rtol=0, atol=1e-9)


# Not testable for layout (the matrices are symmetric by construction):
#   satstate.cov (covariance) and gravity_and_partials (the gravity gradient
#   is the Hessian of the potential). A transposition there is harmless.
