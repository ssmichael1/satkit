#!/usr/bin/env python3
"""Generate SVG plots for MkDocs documentation using matplotlib + SciencePlots.

    python docs/examples/gen_plots.py

Writes SVG files to docs/images/.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import scienceplots  # noqa: F401

plt.style.use(["science", "no-latex"])

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.formatter.use_mathtext": True,
    "svg.fonttype": "none",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.prop_cycle": plt.cycler(color=[
        "#0077BB", "#EE7733", "#009988", "#CC3311",
        "#33BBEE", "#EE3377", "#BBBBBB",
    ]),
})

IMAGES = Path(__file__).resolve().parent.parent / "images"

COLORS = [
    "#0077BB",  # blue
    "#EE7733",  # orange
    "#009988",  # teal
    "#CC3311",  # red
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#BBBBBB",  # grey
]


def savefig(fig, name):
    path = IMAGES / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ── Air Density vs Solar Cycle ─────────────────────────────────────────────

def make_density_plot():
    import satkit as sk

    start = sk.time(1995, 1, 1)
    end = sk.time(2022, 12, 31)
    duration = end - start
    timearray = [start + sk.duration(days=x)
                 for x in np.linspace(0, duration.days, 4000)]

    rho_400 = [sk.density.nrlmsise(400e3, 0, 0, x)[0] for x in timearray]
    rho_500 = [sk.density.nrlmsise(500e3, 0, 0, x)[0] for x in timearray]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    dates = [t.as_datetime() for t in timearray]
    ax.semilogy(dates, rho_400, color=COLORS[0], linewidth=1,
                label="Altitude = 400 km")
    ax.semilogy(dates, rho_500, color=COLORS[1], linewidth=1,
                linestyle="--", label="Altitude = 500 km")
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Density [kg/m$^3$]")
    ax.set_title("Air Density Changes with Solar Cycle")
    ax.legend()
    fig.autofmt_xdate()
    savefig(fig, "density_vs_solar_cycle")


# ── Satellite Forces vs Altitude ───────────────────────────────────────────

def make_force_plot():
    import satkit as sk
    import math as m

    N = 1000
    range_arr = np.logspace(m.log10(6378.2e3 + 100e3), m.log10(50e6), N)

    grav1v = np.array([sk.gravity(np.array([a, 0, 0]), order=1)
                       for a in range_arr])
    grav1 = np.linalg.norm(grav1v, axis=1)

    grav2v = np.array([sk.gravity(np.array([a, 0, 0]), order=2)
                       for a in range_arr])
    grav2 = np.linalg.norm(grav2v - grav1v, axis=1)

    grav6v = np.array([sk.gravity(np.array([a, 0, 0]), order=6)
                       for a in range_arr])
    grav5v = np.array([sk.gravity(np.array([a, 0, 0]), order=5)
                       for a in range_arr])
    grav6 = np.linalg.norm(grav6v - grav5v, axis=1)

    aoverm = 0.01
    Cd = 2.2
    Cr = 1.0

    didx = np.argwhere(range_arr - sk.consts.earth_radius < 800e3).flatten()
    tm_max = sk.time(2001, 12, 1)
    tm_min = sk.time(1996, 12, 1)
    rho_max = np.array([sk.density.nrlmsise(a - sk.consts.earth_radius, 0, 0, tm_max)[0]
                        for a in range_arr[didx]])
    rho_min = np.array([sk.density.nrlmsise(a - sk.consts.earth_radius, 0, 0, tm_min)[0]
                        for a in range_arr[didx]])
    varr = np.sqrt(sk.consts.mu_earth / (range_arr + sk.consts.earth_radius))
    drag_max = 0.5 * rho_max * varr[didx]**2 * Cd * aoverm
    drag_min = 0.5 * rho_min * varr[didx]**2 * Cd * aoverm

    moon_range = np.linalg.norm(
        sk.jplephem.geocentric_pos(sk.solarsystem.Moon, sk.time(2023, 1, 1)))
    moon = sk.consts.mu_moon * (
        (moon_range - range_arr)**(-2) - moon_range**(-2))
    sun = sk.consts.mu_sun * (
        (sk.consts.au - range_arr)**(-2) - sk.consts.au**(-2))

    a_radiation = 4.56e-6 * 0.5 * Cr * aoverm * np.ones(range_arr.shape)

    def add_line(ax, x, y, text, frac=0.5, dx=-20, dy=-20):
        ax.loglog(x, y, "k-", linewidth=1.5)
        idx = int(len(x) * frac)
        ax.annotate(text, xy=(x[idx], y[idx]), fontsize=10,
                    xytext=(dx, dy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="gray"))

    fig, ax = plt.subplots(figsize=(8, 8))
    add_line(ax, range_arr / 1e3, grav1 / 1e3, "Gravity")
    add_line(ax, range_arr / 1e3, grav2 / 1e3, "J2", 0.2, 0, -15)
    add_line(ax, range_arr / 1e3, grav6 / 1e3, "J6", 0.8, 0, -15)
    add_line(ax, range_arr[didx] / 1e3, drag_max / 1e3,
             "Drag Max", 0.7, 30, 0)
    add_line(ax, range_arr[didx] / 1e3, drag_min / 1e3,
             "Drag Min", 0.8, 10, 30)
    add_line(ax, range_arr / 1e3, moon / 1e3, "Moon", 0.8, -10, -15)
    add_line(ax, range_arr / 1e3, sun / 1e3, "Sun", 0.7, -10, 15)
    add_line(ax, range_arr / 1e3, a_radiation / 1e3,
             "Radiation\nPressure", 0.3, -10, 15)
    ax.set_xlabel("Distance from Earth Origin [km]")
    ax.set_ylabel(r"Acceleration [km/s$^2$]")
    ax.set_title("Satellite Forces vs Altitude")
    ax.set_xlim(6378.1, 50e3)
    savefig(fig, "force_vs_altitude")


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    print("Generating SVG plots...")
    make_density_plot()
    make_force_plot()
    print("Done.")
