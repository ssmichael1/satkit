#!/usr/bin/env python3
"""
Validate satkit's ECOM solar-radiation-pressure model against real GPS orbits.

Fits an initial state plus SRP coefficients to a few days of IGS final SP3
positions, then propagates for ~a month with the fitted coefficients and
reports the error growth against the SP3 truth. Three SRP variants are
compared:

  cannonball   - the classic Cr*A/m single coefficient
  ecom-reduced - reduced ECOM1: D0, Y0, B0, Bc, Bs (CODE's operational GPS set)
  ecom2        - ECOM2: D0, Y0, B0, B1c, B1s, D2c, D2s, D4c, D4s
  ecom1        - full ECOM1 (9 once-per-rev terms in u); fits slightly better
                 but predicts ~2x worse than the reduced set, so it is not in
                 the default list

Daily IGS final combined products (15-min, ITRF, centre of mass) are fetched
from BKG (https://igs.bkg.bund.de/root_ftp/IGS/products/<gpsweek>/) - no
login required - and cached locally.

Examples::

    # one PRN: fit 3 days, predict 30, plot error vs time
    python python/examples/ecom_gps_validation.py --prn 20 --start 2024-01-01 \\
        --fit-days 3 --prop-days 30 --cache-dir /tmp/sp3 --plot g20.png

    # 24-hour prediction benchmark over several PRNs (2-day fits)
    python python/examples/ecom_gps_validation.py --start 2024-01-01 \\
        --prns 20 5 1 14 18 11 25 30 --fit-days 2 --prop-days 1 --cache-dir /tmp/sp3

Notes on the setup (each of these was found the hard way):

* SP3 epochs are **GPS time** (header ``%c ... GPS``), 18 s ahead of UTC in
  2024. Reading them as UTC rotates the ITRF->GCRF truth by 18 s of Earth
  rotation relative to the Sun/Moon geometry and quadruples the fit residual.
* The finite-difference Jacobian uses **absolute** steps. scipy's
  ``diff_step`` is relative and silently falls back to sqrt(eps) for
  zero-valued parameters, which is below integrator noise for the harmonic
  ECOM coefficients.
* Arcs that contain Earth-shadow passes use the Gauss-Jackson 8 integrator:
  the adaptive Runge-Kutta steppers can abort with "too many consecutive step
  rejections" at a shadow boundary.

A 3-day fit plus a 30-day prediction runs in a few seconds.
"""

import argparse
import datetime as dt
import gzip
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import requests
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative

import satkit as sk

GPS_EPOCH = dt.date(1980, 1, 6)
BKG = "https://igs.bkg.bund.de/root_ftp/IGS/products"


# --------------------------------------------------------------------------
# SP3 handling
# --------------------------------------------------------------------------
def gps_week(day: dt.date) -> int:
    return (day - GPS_EPOCH).days // 7


def fetch_sp3(day: dt.date, cache: Path) -> Path:
    """Download (once) the IGS final combined SP3 for `day` into `cache`."""
    doy = day.timetuple().tm_yday
    name = f"IGS0OPSFIN_{day.year}{doy:03d}0000_01D_15M_ORB.SP3"
    out = cache / name
    if out.exists():
        return out
    url = f"{BKG}/{gps_week(day)}/{name}.gz"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    cache.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.decompress(r.content))
    return out


def read_sp3(path: Path, prn: int):
    """Return (times, positions_itrf_m) for one GPS PRN from an SP3 file.

    Epochs are GPS time. Epochs flagged as bad/missing (position 0.0 or the
    999999.999999 sentinel) are dropped.
    """
    times, pos = [], []
    current = None
    with open(path) as fd:
        for line in fd:
            if line.startswith("*"):
                y, mo, d, h, mi = (int(line[3:7]), int(line[8:10]), int(line[11:13]),
                                   int(line[14:16]), int(line[17:19]))
                s = float(line[20:31])
                current = sk.time(y, mo, d, h, mi, s, scale=sk.timescale.GPS)
            elif line.startswith(f"PG{prn:02d}") and current is not None:
                x, y_, z = (float(line[4:18]), float(line[18:32]), float(line[32:46]))
                if any(abs(v) > 900000.0 for v in (x, y_, z)) or (x == 0.0 and y_ == 0.0):
                    continue
                times.append(current)
                pos.append([x, y_, z])
    return times, np.array(pos) * 1e3


def load_truth(start: dt.date, ndays: int, prn: int, cache: Path):
    """Concatenate `ndays` daily files into GCRF truth (times, pos_m)."""
    times, pos = [], []
    for i in range(ndays):
        t, p = read_sp3(fetch_sp3(start + dt.timedelta(days=i), cache), prn)
        # Some products repeat the 00:00 epoch of the next day; drop duplicates.
        for tt, pp in zip(t, p):
            if times and tt == times[-1]:
                continue
            times.append(tt)
            pos.append(pp)
    pos = np.array(pos)
    gcrf = np.array([sk.frametransform.qitrf2gcrf(t) * p for t, p in zip(times, pos)])
    return times, gcrf


def eclipse_fraction(times, truth):
    """Fraction of epochs in (cylindrical) Earth shadow."""
    n_shadow = 0
    for t, p in zip(times, truth):
        sun = sk.sun.pos_gcrf(t)
        s_hat = sun / np.linalg.norm(sun)
        along = p @ s_hat
        perp = np.linalg.norm(p - along * s_hat)
        if along < 0 and perp < sk.consts.earth_radius:
            n_shadow += 1
    return n_shadow / len(times)


def beta_angle(pos, vel, t):
    """Sun elevation above the orbit plane (deg)."""
    h = np.cross(pos, vel)
    h /= np.linalg.norm(h)
    s = sk.sun.pos_gcrf(t)
    return np.degrees(np.arcsin(h @ (s / np.linalg.norm(s))))


# --------------------------------------------------------------------------
# Propagation / fitting
# --------------------------------------------------------------------------
def gps_settings(degree: int = 12, eclipses: bool = False):
    s = sk.propsettings()
    s.gravity_degree = degree
    s.gravity_order = degree
    s.use_sun_gravity = True
    s.use_moon_gravity = True
    s.tide_model = sk.tidemodel.solid_step1  # EGM96 (default) is tide-free: consistent with Step 1
    s.use_relativistic_correction = True
    s.use_spaceweather = False
    s.abs_error = 1e-11
    s.rel_error = 1e-11
    s.enable_interp = True
    if eclipses:
        # Adaptive RK steppers can abort at a shadow boundary ("too many
        # consecutive step rejections"); the fixed-step multistep is immune.
        s.integrator = sk.integrator.gauss_jackson8
        s.gj_step_seconds = 60.0
    return s


def _ecom(kind, c):
    c = np.asarray(c, float) * 1e-9  # nm/s^2 -> m/s^2
    return {"ecom-reduced": sk.ecomparams.reduced, "ecom1": sk.ecomparams.ecom1, "ecom2": sk.ecomparams.ecom2}[kind](*c)


VARIANTS = {
    # name: (initial coeffs, absolute FD step per coeff, builder)
    "cannonball": ([0.02], [1e-3], lambda c: sk.satproperties(craoverm=float(c[0]))),
    "ecom-reduced": ([-100.0, 0, 0, 0, 0], [0.1] * 5,
                     lambda c: sk.satproperties(craoverm=0.0, ecom=_ecom("ecom-reduced", c))),
    "ecom2": ([-100.0] + [0.0] * 8, [0.1] * 9,
              lambda c: sk.satproperties(craoverm=0.0, ecom=_ecom("ecom2", c))),
    "ecom1": ([-100.0] + [0.0] * 8, [0.1] * 9,
              lambda c: sk.satproperties(craoverm=0.0, ecom=_ecom("ecom1", c))),
}
DEFAULT_VARIANTS = ["cannonball", "ecom-reduced", "ecom2"]


def propagate_states(x, build, settings, t0, t1, at_times):
    """Propagate [x,y,z (km), vx,vy,vz (m/s), coeffs...]; sample at `at_times`."""
    st = np.concatenate((x[:3] * 1e3, x[3:6]))
    res = sk.propagate(st, t0, t1, propsettings=settings, satproperties=build(x[6:]))
    return np.array(res.interp(at_times))


def initial_velocity(times, truth):
    """O(dt^4) one-sided stencil from the first five positions (the two-point
    chord is ~250 m/s off over a 15-min step at GPS altitude)."""
    h = (times[1] - times[0]).seconds
    p = truth
    return (-25 * p[0] + 48 * p[1] - 36 * p[2] + 16 * p[3] - 3 * p[4]) / (12 * h)


def fit(variant: str, times, truth, settings):
    """Least-squares fit of [state (km, m/s), coeffs] to the truth positions.

    Returns (x, rms_3d_m, seconds). Absolute finite-difference steps: 1 m,
    1e-4 m/s, and 0.1 nm/s^2 (ECOM) or 1e-3 m^2/kg (Cr A/m).
    """
    c0, cstep, build = VARIANTS[variant]
    t0, t1 = times[0], times[-1]
    x0 = np.concatenate((truth[0] / 1e3, initial_velocity(times, truth), c0))
    steps = np.array([1e-3] * 3 + [1e-4] * 3 + cstep)

    def resid(x):
        return (propagate_states(x, build, settings, t0, t1, times)[:, :3] - truth).ravel()

    tic = _time.perf_counter()
    sol = least_squares(resid, x0, jac=lambda x: approx_derivative(resid, x, abs_step=steps),
                        x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=400)
    rms3d = np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 3) ** 2, axis=1)))
    return sol.x, rms3d, _time.perf_counter() - tic


def rtn_components(pos, vel, dvec):
    r_hat = pos / np.linalg.norm(pos)
    n_hat = np.cross(pos, vel)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([dvec @ r_hat, dvec @ t_hat, dvec @ n_hat])


def evaluate(x, variant, settings, times, truth):
    """Propagate the fitted solution over all `times`; return per-epoch errors."""
    build = VARIANTS[variant][2]
    states = propagate_states(x, build, settings, times[0], times[-1], times)
    d = states[:, :3] - truth
    err3d = np.linalg.norm(d, axis=1)
    rtn = np.array([rtn_components(s[:3], s[3:], dd) for s, dd in zip(states, d)])
    days = np.array([(t - times[0]).days for t in times])  # duration.days is fractional
    return days, err3d, rtn


def day_rms(err, days, day):
    m = (days >= day - 1) & (days < day)
    return (np.sqrt(np.mean(err[m] ** 2)), err[m].max()) if m.any() else (np.nan, np.nan)


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def run_single(args):
    ndays = args.prop_days + 1  # +1 so the final day's last epoch is covered
    times, truth = load_truth(args.start, ndays, args.prn, args.cache_dir)
    t_fit_end = times[0] + sk.duration.from_days(args.fit_days)
    n_fit = sum(1 for t in times if t <= t_fit_end)
    ecl = eclipse_fraction(times, truth)
    print(f"PRN G{args.prn:02d}: {len(times)} SP3 epochs from {times[0]} to {times[-1]} (GPS time); "
          f"fitting on first {n_fit} ({args.fit_days} d), evaluating {args.prop_days} d")
    print(f"eclipse: {ecl * 100:.1f}% of epochs in Earth shadow "
          f"({'eclipse season - using Gauss-Jackson 8' if ecl > 0 else 'no eclipses in window'})")

    settings = gps_settings(args.degree, eclipses=ecl > 0)
    checkpoints = [1, 3, 7, 14, 21, 30]
    results = {}
    for variant in args.variants:
        x, rms, secs = fit(variant, times[:n_fit], truth[:n_fit], settings)
        unit = "m^2/kg" if variant == "cannonball" else "nm/s^2"
        print(f"\n[{variant}] fit RMS {rms:.3f} m (3D) over {args.fit_days} d ({secs:.1f} s); coefficients ({unit}): "
              + ", ".join(f"{c:.3f}" for c in x[6:]))
        days, err, rtn = evaluate(x, variant, settings, times, truth)
        results[variant] = (days, err, rtn)
        print(f"  {'day':>4} {'3D RMS (m)':>11} {'3D max (m)':>11}   {'R rms':>8} {'T rms':>8} {'N rms':>8}")
        for cp in checkpoints:
            if cp > args.prop_days:
                break
            m = (days >= cp - 1) & (days < cp)
            if not m.any():
                continue
            rr = np.sqrt(np.mean(rtn[m] ** 2, axis=0))
            print(f"  {cp:>4} {np.sqrt(np.mean(err[m] ** 2)):>11.3f} {err[m].max():>11.3f}   "
                  f"{rr[0]:>8.3f} {rr[1]:>8.3f} {rr[2]:>8.3f}")
        above = np.nonzero(err[n_fit:] > 10.0)[0]
        if above.size:
            print(f"  first epoch above 10 m: day {days[n_fit + above[0]]:.1f}")
        else:
            print(f"  never above 10 m within {args.prop_days} d")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for variant, (days, err, _) in results.items():
            ax.plot(days, err, label=variant, lw=0.8)
        ax.axvline(args.fit_days, color="k", ls="--", lw=0.8, label="end of fit window")
        ax.axhline(10, color="r", ls=":", lw=0.8, label="10 m")
        ax.set_yscale("log")
        ax.set_xlabel("days since epoch")
        ax.set_ylabel("3D position error vs IGS final (m)")
        ax.set_title(f"GPS G{args.prn:02d} from {args.start}: satkit propagation with fitted SRP")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nsaved {args.plot}")


def run_benchmark(args):
    """`--fit-days`-day fit -> `--prop-days`-day prediction for several PRNs."""
    ndays = args.fit_days + args.prop_days + 1
    print(f"{'PRN':>4} {'beta':>6} {'shadow':>6}  " + "  ".join(f"{v:>23s}" for v in args.variants))
    print(f"{'':>4} {'(deg)':>6} {'(%)':>6}  " + "  ".join(f"{'fit RMS / pred RMS (m)':>23s}" for _ in args.variants))
    table = {v: [] for v in args.variants}
    for prn in args.prns:
        try:
            times, truth = load_truth(args.start, ndays, prn, args.cache_dir)
        except Exception as e:  # missing satellite / download problem
            print(f"G{prn:02d}: skipped ({e})")
            continue
        t_fit_end = times[0] + sk.duration.from_days(args.fit_days)
        n_fit = sum(1 for t in times if t <= t_fit_end)
        ecl = eclipse_fraction(times, truth)
        settings = gps_settings(args.degree, eclipses=ecl > 0)
        v0 = initial_velocity(times, truth)
        beta = beta_angle(truth[0], v0, times[0])
        row = []
        for variant in args.variants:
            try:
                x, rms, _ = fit(variant, times[:n_fit], truth[:n_fit], settings)
                days, err, _ = evaluate(x, variant, settings, times, truth)
                pred, _ = day_rms(err, days, args.fit_days + args.prop_days)
                table[variant].append(pred)
                row.append(f"{rms:>10.3f} / {pred:>10.3f}")
            except RuntimeError as e:
                row.append(f"{'failed':>23s}")
                print(f"  G{prn:02d} {variant}: {e}")
        print(f"G{prn:02d} {beta:>6.1f} {ecl * 100:>6.1f}  " + "  ".join(row))
    print(f"{'median':>18}  " + "  ".join(f"{'':>10}   {np.median(table[v]):>10.3f}" if table[v] else f"{'-':>23s}" for v in args.variants))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prn", type=int, default=20, help="GPS PRN for the single-satellite run (default 20)")
    ap.add_argument("--prns", type=int, nargs="*", default=None,
                    help="run the multi-satellite benchmark over these PRNs instead")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1), help="first day (YYYY-MM-DD)")
    ap.add_argument("--fit-days", type=int, default=3)
    ap.add_argument("--prop-days", type=int, default=30,
                    help="days to evaluate after the epoch (single run) or after the fit arc (benchmark)")
    ap.add_argument("--degree", type=int, default=12, help="gravity degree/order")
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS, choices=list(VARIANTS))
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("SP3_CACHE", "sp3-cache")))
    ap.add_argument("--plot", type=Path, default=None, help="save a PNG of error vs time (single run)")
    args = ap.parse_args()
    if args.prns:
        run_benchmark(args)
    else:
        run_single(args)


if __name__ == "__main__":
    sys.exit(main())
