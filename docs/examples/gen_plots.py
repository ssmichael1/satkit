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

plt.style.use(["science", "no-latex",
    str(Path(__file__).resolve().parent.parent / "satkit.mplstyle")])

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


# ── Satellite-Local Frame Diagrams ─────────────────────────────────────────

def _orbit_state(a_m: float, e: float, nu_rad: float):
    """Compute (r_m, v_ms) in the perifocal plane for a Kepler orbit with
    semi-major axis a, eccentricity e, at true anomaly nu. Perigee is on
    the +x axis; motion is counter-clockwise (angular momentum out of page).
    """
    mu = 3.986004418e14  # m^3 / s^2
    p = a_m * (1.0 - e * e)
    r_mag = p / (1.0 + e * np.cos(nu_rad))
    pos = r_mag * np.array([np.cos(nu_rad), np.sin(nu_rad), 0.0])
    k = np.sqrt(mu / p)
    v_r = k * e * np.sin(nu_rad)
    v_theta = k * (1.0 + e * np.cos(nu_rad))
    v_x = -v_theta * np.sin(nu_rad) + v_r * np.cos(nu_rad)
    v_y = v_theta * np.cos(nu_rad) + v_r * np.sin(nu_rad)
    vel = np.array([v_x, v_y, 0.0])
    return pos, vel


def _ric_axes(pos, vel):
    """Return (R, I, C) unit vectors in the parent (perifocal/GCRF) frame."""
    r_hat = pos / np.linalg.norm(pos)
    h = np.cross(pos, vel)
    h_hat = h / np.linalg.norm(h)
    i_hat = np.cross(h_hat, r_hat)
    return r_hat, i_hat, h_hat


def _ntw_axes(pos, vel):
    """Return (N, T, W) unit vectors."""
    t_hat = vel / np.linalg.norm(vel)
    h = np.cross(pos, vel)
    w_hat = h / np.linalg.norm(h)
    n_hat = np.cross(t_hat, w_hat)
    return n_hat, t_hat, w_hat


def _lvlh_axes(pos, vel):
    """Return (x, y, z) LVLH unit vectors."""
    r_hat = pos / np.linalg.norm(pos)
    h = np.cross(pos, vel)
    h_hat = h / np.linalg.norm(h)
    z_hat = -r_hat
    y_hat = -h_hat
    x_hat = np.cross(y_hat, z_hat)
    return x_hat, y_hat, z_hat


def _draw_orbit(ax, a_m, e):
    """Draw an ellipse orbit with focus at the origin, perigee on +x."""
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    p = a_m * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x / 1e6, y / 1e6, color="#aaaaaa", linewidth=0.8, zorder=1)
    # Earth at origin
    earth_r = 6.378e6
    circ = plt.Circle((0, 0), earth_r / 1e6, color="#4a7a9c",
                      alpha=0.5, zorder=0)
    ax.add_patch(circ)


def _draw_axis(ax, origin, direction, length, label, color, offset=(0.0, 0.0)):
    """Draw a labeled axis arrow from `origin` of given length."""
    end = origin + direction * length
    ax.annotate(
        "", xy=(end[0] / 1e6, end[1] / 1e6),
        xytext=(origin[0] / 1e6, origin[1] / 1e6),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=15),
        zorder=5,
    )
    ax.text(
        end[0] / 1e6 + offset[0], end[1] / 1e6 + offset[1],
        label, color=color, fontsize=11, fontweight="bold",
        ha="center", va="center", zorder=6,
    )


def _draw_cross_track_marker(ax, pos, label, color, offset):
    """Draw a ⊙ symbol (vector out of page) at the satellite position."""
    cx, cy = pos[0] / 1e6, pos[1] / 1e6
    ax.scatter([cx], [cy], s=160, facecolors="white",
               edgecolors=color, linewidths=1.5, zorder=6)
    ax.scatter([cx], [cy], s=18, color=color, zorder=7)
    ax.text(cx + offset[0], cy + offset[1], label,
            color=color, fontsize=10, fontweight="bold",
            ha="center", va="center", zorder=6)


def _frame_overlay(ax, pos, vel, arrow_len, title):
    """Overlay RTN, NTW, LVLH in-plane axes on the given orbit state.

    All three frames share the same cross-track axis (up to sign), so we
    mark it only once per frame with an offset to keep labels legible.
    """
    col_ric = COLORS[0]   # blue
    col_ntw = COLORS[1]   # orange
    col_lvlh = COLORS[3]  # red

    r_hat, i_hat, _ = _ric_axes(pos, vel)
    n_hat, t_hat, _ = _ntw_axes(pos, vel)
    x_hat, _, z_hat = _lvlh_axes(pos, vel)

    # Mark satellite
    ax.scatter([pos[0] / 1e6], [pos[1] / 1e6], s=28, color="k", zorder=5)

    # RTN: R and I
    _draw_axis(ax, pos, r_hat, arrow_len, r"$\hat R$", col_ric,
               offset=(0.25, 0.25))
    _draw_axis(ax, pos, i_hat, arrow_len, r"$\hat T$", col_ric,
               offset=(0.25, 0.25))
    # NTW: N and T
    _draw_axis(ax, pos, n_hat, arrow_len, r"$\hat N$", col_ntw,
               offset=(-0.35, 0.0))
    _draw_axis(ax, pos, t_hat, arrow_len, r"$\hat T$", col_ntw,
               offset=(0.35, 0.0))
    # LVLH: x and -z (we omit z=-R̂ to avoid clutter — LVLH x is what differs)
    _draw_axis(ax, pos, x_hat, arrow_len * 0.75, r"$\hat x_{LVLH}$", col_lvlh,
               offset=(0.15, -0.45))

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [Mm]")
    ax.set_ylabel("y [Mm]")
    ax.grid(True, alpha=0.3)


def make_frame_diagrams():
    """Three figures visualizing RTN / NTW / LVLH axes and the flight-path
    angle gap on eccentric orbits. Referenced from docs/guide/maneuver_frames.md.
    """

    # ── Figure 1: Circular orbit — all three frames coincide ────────────
    pos_c, vel_c = _orbit_state(a_m=7.0e6, e=0.0, nu_rad=np.deg2rad(50.0))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    _draw_orbit(ax, 7.0e6, 0.0)
    _frame_overlay(
        ax, pos_c, vel_c, arrow_len=1.8e6,
        title=r"Circular orbit ($\gamma=0$): RTN, NTW, LVLH axes coincide",
    )
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=COLORS[0], lw=2, label="RTN"),
            plt.Line2D([], [], color=COLORS[1], lw=2, label="NTW"),
            plt.Line2D([], [], color=COLORS[3], lw=2, label="LVLH"),
        ],
        loc="upper left", fontsize=10, framealpha=0.9,
    )
    savefig(fig, "frames_circular")

    # ── Figure 2: Eccentric orbit at mid-anomaly — NTW and RTN diverge ──
    # Note: the guide's numeric example (γ ≈ 12.7°) depends only on (e, ν),
    # not on a, so we use a larger a than the prose to keep the orbit well
    # outside the Earth disk for visualization.
    a = 1.5e7
    e = 0.3
    nu = np.deg2rad(60.0)
    pos_e, vel_e = _orbit_state(a_m=a, e=e, nu_rad=nu)
    gamma = np.arctan(e * np.sin(nu) / (1.0 + e * np.cos(nu)))
    gamma_deg = np.rad2deg(gamma)

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_orbit(ax, a, e)
    _frame_overlay(
        ax, pos_e, vel_e, arrow_len=3.5e6,
        title=(r"Eccentric orbit ($e=0.3$, $\nu=60°$): "
               rf"$\gamma={gamma_deg:.1f}°$"),
    )
    ax.set_xlim(-22, 14)
    ax.set_ylim(-16, 18)

    # Annotate the flight-path angle between R̂ and N̂ (or equivalently
    # between Î and T̂)
    r_hat, i_hat, _ = _ric_axes(pos_e, vel_e)
    n_hat, t_hat, _ = _ntw_axes(pos_e, vel_e)
    # Mid-angle arc between I and T
    ang_i = np.arctan2(i_hat[1], i_hat[0])
    ang_t = np.arctan2(t_hat[1], t_hat[0])
    arc_r = 1.2
    arc_theta = np.linspace(min(ang_i, ang_t), max(ang_i, ang_t), 40)
    arc_x = pos_e[0] / 1e6 + arc_r * np.cos(arc_theta)
    arc_y = pos_e[1] / 1e6 + arc_r * np.sin(arc_theta)
    ax.plot(arc_x, arc_y, "k-", linewidth=1.2, zorder=4)
    mid_theta = 0.5 * (ang_i + ang_t)
    lx = pos_e[0] / 1e6 + 1.6 * np.cos(mid_theta)
    ly = pos_e[1] / 1e6 + 1.6 * np.sin(mid_theta)
    ax.text(lx, ly, rf"$\gamma$", fontsize=12, ha="center", va="center")

    ax.legend(
        handles=[
            plt.Line2D([], [], color=COLORS[0], lw=2, label="RTN"),
            plt.Line2D([], [], color=COLORS[1], lw=2, label="NTW"),
            plt.Line2D([], [], color=COLORS[3], lw=2, label="LVLH"),
        ],
        loc="upper left", fontsize=10, framealpha=0.9,
    )
    savefig(fig, "frames_eccentric")

    # ── Figure 3: Burn-direction consequence — NTW +T vs RTN +T ──────────
    # Zoom in on the velocity vector and show how a 10 m/s delta-v in each
    # frame produces different |v| changes.
    fig, ax = plt.subplots(figsize=(7, 6))

    # Origin at the satellite (velocity-space view)
    # Scale: draw |v_before| as length 1 unit, delta-v arrows as ~0.15 units
    v_before = vel_e
    v_mag = np.linalg.norm(v_before)

    dv_mag = 1500.0  # m/s, exaggerated for visibility (real burns are tiny)
    _, _, ric_c = _ric_axes(pos_e, vel_e)
    _, t_unit, _ = _ntw_axes(pos_e, vel_e)
    _, i_unit, _ = _ric_axes(pos_e, vel_e)

    # NTW +T: delta-v along velocity
    dv_ntw = t_unit * dv_mag
    v_after_ntw = v_before + dv_ntw

    # RTN +T: delta-v along I axis (perpendicular to R, in orbit plane)
    dv_ric = i_unit * dv_mag
    v_after_ric = v_before + dv_ric

    # Draw the three velocity vectors from a common origin
    def arr(ax, start, end, color, lw=2.0, ls="-"):
        ax.annotate(
            "", xy=(end[0], end[1]), xytext=(start[0], start[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                            linestyle=ls, mutation_scale=18),
        )

    origin = np.array([0.0, 0.0])
    scale = 1.0 / 1000.0  # m/s -> km/s for plot units

    v_plot = v_before * scale
    arr(ax, origin, v_plot, "k", lw=2.5)
    ax.text(v_plot[0] * 0.55, v_plot[1] * 0.55 - 0.3,
            rf"$\vec v$ (|v| = {v_mag / 1000:.2f} km/s)",
            fontsize=10, color="k")

    # NTW delta-v: tail at tip of v_before, along v̂
    arr(ax, v_plot, (v_before + dv_ntw) * scale, COLORS[1], lw=2.5)
    ax.text(
        (v_before + 0.6 * dv_ntw)[0] * scale + 0.3,
        (v_before + 0.6 * dv_ntw)[1] * scale + 0.3,
        r"NTW $+\hat T$", fontsize=10, color=COLORS[1], fontweight="bold",
    )

    # RTN delta-v: tail at tip of v_before, along t̂_rtn
    arr(ax, v_plot, (v_before + dv_ric) * scale, COLORS[0], lw=2.5)
    ax.text(
        (v_before + 0.6 * dv_ric)[0] * scale - 1.2,
        (v_before + 0.6 * dv_ric)[1] * scale - 0.5,
        r"RTN $+\hat T$", fontsize=10, color=COLORS[0], fontweight="bold",
    )

    # Dashed circles showing |v_after| magnitudes
    for v_after, color, label in [
        (v_after_ntw, COLORS[1], "|v| after NTW"),
        (v_after_ric, COLORS[0], "|v| after RTN"),
    ]:
        mag = np.linalg.norm(v_after) * scale
        theta_c = np.linspace(0, 2 * np.pi, 200)
        ax.plot(mag * np.cos(theta_c), mag * np.sin(theta_c),
                color=color, linestyle=":", linewidth=1.0, alpha=0.6)

    # Also dashed circle for |v_before|
    mag0 = v_mag * scale
    theta_c = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mag0 * np.cos(theta_c), mag0 * np.sin(theta_c),
            color="k", linestyle=":", linewidth=0.8, alpha=0.4)

    # Compute the actual |v| changes to report
    delta_ntw = np.linalg.norm(v_after_ntw) - v_mag
    delta_ric = np.linalg.norm(v_after_ric) - v_mag
    info = (rf"$\Delta v = {dv_mag / 1000:.1f}$ km/s along each frame's"
            "\nprograde axis\n\n"
            rf"NTW $+\hat T$: $|v|$ gains {delta_ntw / 1000:.3f} km/s"
            rf" (= $\Delta v$)"
            "\n"
            rf"RTN $+\hat T$: $|v|$ gains {delta_ric / 1000:.3f} km/s"
            rf" (= $\Delta v \cos\gamma$)")
    ax.text(
        0.02, 0.98, info,
        transform=ax.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#bbbbbb", alpha=0.95),
    )

    ax.set_aspect("equal")
    ax.set_xlabel(r"$v_x$ [km/s]")
    ax.set_ylabel(r"$v_y$ [km/s]")
    ax.set_title(
        rf"Velocity-space view at $e=0.3$, $\nu=60°$ ($\gamma={gamma_deg:.1f}°$)"
    )
    ax.grid(True, alpha=0.3)

    # Set tight axis limits around the interesting region
    all_x = [0, v_plot[0], (v_before + dv_ntw)[0] * scale,
             (v_before + dv_ric)[0] * scale]
    all_y = [0, v_plot[1], (v_before + dv_ntw)[1] * scale,
             (v_before + dv_ric)[1] * scale]
    xpad = 2.0
    ax.set_xlim(min(all_x) - xpad, max(all_x) + xpad)
    ax.set_ylim(min(all_y) - xpad, max(all_y) + xpad)
    savefig(fig, "frames_burn_comparison")


if __name__ == "__main__":
    os.makedirs(IMAGES, exist_ok=True)
    print("Generating SVG plots...")
    make_density_plot()
    make_force_plot()
    make_frame_diagrams()
    print("Done.")
