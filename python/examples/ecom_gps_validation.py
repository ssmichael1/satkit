#!/usr/bin/env python3
"""
Validate satkit's ECOM solar-radiation-pressure model against real GPS orbits.

Fits an initial state plus SRP coefficients to a few days of IGS final SP3
positions, then propagates for ~a month with the fitted coefficients and
reports the error growth against the SP3 truth. Three SRP variants are
compared:

  cannonball  - the classic Cr*A/m single coefficient
  ecom-reduced - reduced ECOM1: D0, Y0, B0, Bc, Bs (CODE's operational GPS set)
  ecom2        - ECOM2: D0, Y0, B0, B1c, B1s, D2c, D2s, D4c, D4s

Daily IGS final combined products (15-min, ITRF, centre of mass) are fetched
from BKG (https://igs.bkg.bund.de/root_ftp/IGS/products/<gpsweek>/) — no
login required — and cached locally.

Example::

    python python/examples/ecom_gps_validation.py --prn 20 --start 2024-01-01 \\
        --fit-days 3 --prop-days 30 --cache-dir /tmp/sp3

Runs in a few minutes (dominated by the finite-difference fits).
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

import satkit as sk

GPS_EPOCH = dt.date(1980, 1, 6)
BKG = "https://igs.bkg.bund.de/root_ftp/IGS/products"
PSUN = 4.56e-6  # N/m^2 at 1 AU (same constant as the propagator's cannonball term)


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

    Epochs flagged as bad/missing (position 0.0 or the 999999.999999
    sentinel) are dropped.
    """
    times, pos = [], []
    current = None
    with open(path) as fd:
        for line in fd:
            if line.startswith("*"):
                y, mo, d, h, mi = (int(line[3:7]), int(line[8:10]), int(line[11:13]),
                                   int(line[14:16]), int(line[17:19]))
                s = float(line[20:31])
                current = sk.time(y, mo, d, h, mi, s)
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
        # SP3 daily files include the 00:00 epoch of the next day only in some
        # products; drop exact duplicates at day boundaries.
        for tt, pp in zip(t, p):
            if times and tt == times[-1]:
                continue
            times.append(tt)
            pos.append(pp)
    pos = np.array(pos)
    gcrf = np.array([sk.frametransform.qitrf2gcrf(t) * p for t, p in zip(times, pos)])
    return times, gcrf


# --------------------------------------------------------------------------
# Propagation / fitting
# --------------------------------------------------------------------------
def gps_settings(degree: int = 12):
    s = sk.propsettings()
    s.gravity_degree = degree
    s.gravity_order = degree
    s.use_sun_gravity = True
    s.use_moon_gravity = True
    s.tide_model = sk.tidemodel.solid_step1
    s.use_relativistic_correction = True
    s.use_spaceweather = False
    s.abs_error = 1e-11
    s.rel_error = 1e-11
    s.enable_interp = True
    return s


VARIANTS = {
    # name: (n_coeffs, initial coeffs, diff_step per coeff, builder)
    "cannonball": (1, [0.02], [1e-2], lambda c: sk.satproperties(craoverm=c[0])),
    "ecom-reduced": (
        5, [-100.0, 0, 0, 0, 0], [0.1] * 5,
        lambda c: sk.satproperties(craoverm=0.0, ecom=sk.ecomparams.reduced(*(np.asarray(c) * 1e-9))),
    ),
    "ecom2": (
        9, [-100.0] + [0.0] * 8, [0.1] * 9,
        lambda c: sk.satproperties(craoverm=0.0, ecom=sk.ecomparams.ecom2(*(np.asarray(c) * 1e-9))),
    ),
}


def propagate_positions(state_km_ms, props, settings, t0, t1, times):
    st = np.concatenate((state_km_ms[:3] * 1e3, state_km_ms[3:]))
    res = sk.propagate(st, t0, t1, propsettings=settings, satproperties=props)
    return np.array(res.interp(times))


def fit(variant: str, times, truth, settings):
    """Least-squares fit of [state (km, m/s), coeffs] to the truth positions."""
    ncoef, c0, dstep, build = VARIANTS[variant]
    t0, t1 = times[0], times[-1]
    dt_s = (times[1] - times[0]).seconds
    v0 = (truth[1] - truth[0]) / dt_s
    x0 = np.concatenate((truth[0] / 1e3, v0, c0))

    def resid(x):
        return (propagate_positions(x[:6], build(x[6:]), settings, t0, t1, times)[:, :3] - truth).ravel()

    # Explicit finite-difference steps: scipy's default relative step on a
    # zero-valued coefficient (~1e-8 nm/s^2) is below integrator noise.
    tic = _time.perf_counter()
    sol = least_squares(resid, x0, x_scale="jac", diff_step=[1e-6] * 6 + dstep, max_nfev=400)
    rms = np.sqrt(np.mean(sol.fun**2))
    return sol.x, rms, _time.perf_counter() - tic


def rtn_components(pos, vel, dvec):
    r_hat = pos / np.linalg.norm(pos)
    n_hat = np.cross(pos, vel)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([dvec @ r_hat, dvec @ t_hat, dvec @ n_hat])


def evaluate(x, variant, settings, times, truth, t_fit_end):
    """Propagate the fitted solution over all `times`; return per-epoch errors."""
    _, _, _, build = VARIANTS[variant]
    states = propagate_positions(x[:6], build(x[6:]), settings, times[0], times[-1], times)
    d = states[:, :3] - truth
    err3d = np.linalg.norm(d, axis=1)
    rtn = np.array([rtn_components(s[:3], s[3:], dd) for s, dd in zip(states, d)])
    days = np.array([(t - times[0]).days for t in times])
    return days, err3d, rtn


def eclipse_fraction(times, truth):
    """Fraction of epochs in (partial) Earth shadow, from the Sun geometry."""
    n_shadow = 0
    for t, p in zip(times, truth):
        sun = sk.sun.pos_gcrf(t)
        # Cylindrical shadow test is enough to flag an eclipse season.
        s_hat = sun / np.linalg.norm(sun)
        along = p @ s_hat
        perp = np.linalg.norm(p - along * s_hat)
        if along < 0 and perp < sk.consts.earth_radius:
            n_shadow += 1
    return n_shadow / len(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prn", type=int, default=20, help="GPS PRN (default 20)")
    ap.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2024, 1, 1), help="first day (YYYY-MM-DD)")
    ap.add_argument("--fit-days", type=int, default=3)
    ap.add_argument("--prop-days", type=int, default=30)
    ap.add_argument("--degree", type=int, default=12, help="gravity degree/order")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("SP3_CACHE", "sp3-cache")))
    ap.add_argument("--plot", type=Path, default=None, help="save a PNG of error vs time")
    args = ap.parse_args()

    ndays = args.prop_days + 1  # +1 so the final day's last epoch is covered
    times, truth = load_truth(args.start, ndays, args.prn, args.cache_dir)
    t_fit_end = times[0] + sk.duration.from_days(args.fit_days)
    n_fit = sum(1 for t in times if t <= t_fit_end)
    print(f"PRN G{args.prn:02d}: {len(times)} SP3 epochs from {times[0]} to {times[-1]}; "
          f"fitting on first {n_fit} ({args.fit_days} d), evaluating {args.prop_days} d")
    ecl = eclipse_fraction(times, truth)
    print(f"eclipse: {ecl * 100:.1f}% of epochs in Earth shadow "
          f"({'eclipse season' if ecl > 0 else 'no eclipses in window'})")

    settings = gps_settings(args.degree)
    checkpoints = [1, 3, 7, 14, 21, 30]
    results = {}
    for variant in args.variants:
        x, rms, secs = fit(variant, times[:n_fit], truth[:n_fit], settings)
        coeffs = x[6:]
        unit = "m^2/kg" if variant == "cannonball" else "nm/s^2"
        print(f"\n[{variant}] fit RMS {rms:.3f} m over {args.fit_days} d ({secs:.0f} s); coefficients ({unit}): "
              + ", ".join(f"{c:.3f}" for c in coeffs))
        days, err, rtn = evaluate(x, variant, settings, times, truth, t_fit_end)
        results[variant] = (days, err, rtn)
        print(f"  {'day':>4} {'3D RMS (m)':>11} {'3D max (m)':>11}   {'R rms':>8} {'T rms':>8} {'N rms':>8}")
        for cp in checkpoints:
            if cp > args.prop_days:
                break
            m = (days >= cp - 1) & (days < cp)
            if not m.any():
                continue
            rr = np.sqrt(np.mean(rtn[m] ** 2, axis=0))
            print(f"  {cp:>4} {np.sqrt(np.mean(err[m] ** 2)):>11.2f} {err[m].max():>11.2f}   "
                  f"{rr[0]:>8.2f} {rr[1]:>8.2f} {rr[2]:>8.2f}")
        last = days >= args.prop_days - 1
        print(f"  final-day max 3D error: {err[last].max():.2f} m")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for variant, (days, err, _) in results.items():
            tdays = [(t - times[0]).days for t in times]  # duration.days is fractional
            ax.plot(tdays, err, label=variant, lw=0.8)
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


if __name__ == "__main__":
    sys.exit(main())
