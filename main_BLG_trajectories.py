"""Trajektorie semiclassical snake states dla punktow zadania 9.

Profil n(x) i B(x) oraz geometria trajektorii odpowiadaja abc.py.
Dla kazdego Vb solver zwraca n2 po lewej stronie i n1 po prawej.
"""

from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from BLG_parameters import CapacitanceParameters, BLGPhysicsParameters, DoubleBLGSolver

# d=60
B_VALUES_BY_VB = {
    20.0: (0.49, 0.75, 1.1, 2.2, 2.9, 3.45, 5.0, 6.25, 7.22, 7.7, 8.98, 9.6),
    30.0: (0.48, 0.7, 1.3, 2.18, 4.98, 5.75, 6.3, 6.8, 7.8, 8.85),
    45.0: (0.49, 0.7, 1.22, 2.0, 2.9, 3.85, 4.85, 5.75, 7.6, 9.2),
}

#d=0.001
# B_VALUES_BY_VB = {
#     20.0: (0.10, 0.70, 2.11, 2.91, 3.75, 6.23, 7.71, 8.46, 9.2, 0.34, 1.20, 2.65, 3.15, 4.93, 7.31, 8.18, 8.72, 9.82),
#     30.0: (0.08, 0.62, 2.11, 3.85, 5.99, 7.52,  9.16, 0.32, 1.22, 2.95, 4.75, 7.25, 7.74, 9.82),
#     45.0: (0.30, 1.30, 4.51, 5.81, 6.95, 0.10, 0.62, 3.05, 5.43, 6.15, 8.32),
# }
hbar = 1.054571817e-34
vF = 1.0e6
e_charge = 1.602176634e-19


def n_x(Vx, Ux, gamma1=0.39, hvf2=6.39**2, hbar=None, vF=None):
    """Lokalna gęstość nośników w jednostkach 10^15 m^-2 z V(x), U(x) w eV.

    Ta forma jest zamkniętym odwróceniem tego samego równania co w solverze
    _f4t / _f4b. Działa wektorowo, bez kosztownego brentq dla każdego punktu.
    """
    Vx = np.asarray(Vx, dtype=float)
    Ux = np.asarray(Ux, dtype=float)

    hbar_val = hbar if hbar is not None else 1.054571817e-34
    vF_val = vF if vF is not None else 1.0e6
    hvf2_val = hvf2 if hvf2 is not None else (hbar_val * vF_val) ** 2

    gamma1_j = float(gamma1) * e_charge
    Vj = Vx * e_charge
    Uj = Ux * e_charge

    result = np.zeros_like(Vj, dtype=float)
    mask = np.abs(Vj) > 1e-30
    if np.any(mask):
        Vmask = Vj[mask]
        Umask = Uj[mask]

        rad = np.sqrt(
            np.maximum(
                4.0 * Vmask**2 * gamma1_j**2
                + 4.0 * Vmask**2 * Umask**2
                - gamma1_j**2 * Umask**2,
                0.0,
            )
        )

        n_abs_si = (
            Vmask**2
            + Umask**2 / 4.0
            + 0.5 * rad
        ) / (np.pi * hbar_val**2 * vF_val**2)

        result[mask] = -np.sign(Vmask) * n_abs_si / 1e15
    return result


def tanh_profile(x, left_value, right_value, d_m):
    """Interpolacja tanh pomiędzy lewą i prawą wartością asymptotyczną."""
    return 0.5 * (left_value + right_value) + 0.5 * (right_value - left_value) * np.tanh(x / d_m)


def build_profile(V_left, V_right, U_left, U_right, B0, d_m, x_grid_m):
    """Zbuduj interpolowany profil S(x)=integral(kappa dx) z lokalnym n(x)=n_x(V(x), U(x))."""
    V_grid = tanh_profile(x_grid_m, V_left, V_right, d_m)
    U_grid = tanh_profile(x_grid_m, U_left, U_right, d_m)
    n_grid = n_x(V_grid, U_grid, gamma1=0.39, hvf2=6.39**2)
    B_grid = B0 * np.tanh(x_grid_m / d_m)
    kappa_grid = np.zeros_like(x_grid_m)
    valid = (np.abs(B_grid) > 1e-15) & (np.abs(n_grid) > 1e-30)
    kappa_grid[valid] = (
        1.602176634e-19 * np.abs(B_grid[valid])
        / (1.054571817e-34 * np.sqrt(np.pi * np.abs(n_grid[valid]) * 1e15))
    )

    zero_index = int(np.argmin(np.abs(x_grid_m)))
    S_grid = np.zeros_like(x_grid_m)
    S_grid[zero_index:] = cumulative_trapezoid(
        kappa_grid[zero_index:], x_grid_m[zero_index:], initial=0.0
    )
    S_grid[:zero_index + 1] = cumulative_trapezoid(
        kappa_grid[:zero_index + 1][::-1],
        x_grid_m[:zero_index + 1][::-1],
        initial=0.0,
    )[::-1]

    def S(x):
        return float(np.interp(x, x_grid_m, S_grid))

    return S


def find_turning_point(S, side, search_limit_m):
    x_grid = np.linspace(0.0, search_limit_m, 4000)[1:] * side
    f_prev = abs(S(x_grid[0])) - 1.0
    x_prev = x_grid[0]
    for x in x_grid[1:]:
        f_now = abs(S(x)) - 1.0
        if f_prev * f_now <= 0.0:
            return brentq(lambda xx: abs(S(xx)) - 1.0, x_prev, x,
                          xtol=1e-14, rtol=1e-12)
        x_prev, f_prev = x, f_now
    return None


def calculate_turning_distances(V_left, V_right, U_left, U_right, B0, d_m):
    x_grid = np.linspace(-5e-6, 5e-6, 200001)
    S = build_profile(V_left, V_right, U_left, U_right, B0, d_m, x_grid)
    x_left = find_turning_point(S, -1, 5e-6)
    x_right = find_turning_point(S, +1, 5e-6)
    return x_left, x_right


def trace_trajectory(V_left, V_right, U_left, U_right, B0, d_m, x_min_m, x_max_m,
                     y_min_m, y_max_m, x0_m=0.1e-9, y0_m=-150e-9,
                     theta0=0, ds_m=0.1e-9):
    """Śledź trajektorię z lokalnym n(x)=n_x(V(x),U(x)), a nie z prostym profilem tanh n_left→n_right."""
    x_grid = np.linspace(-5e-6, 5e-6, 200001)
    S = build_profile(V_left, V_right, U_left, U_right, B0, d_m, x_grid)

    x = x0_m
    y = y0_m
    theta = theta0
    xs = []
    ys = []
    max_steps = int(2e7)

    for _ in range(max_steps):
        if not (x_min_m <= x <= x_max_m and y_min_m <= y <= y_max_m):
            break
        xs.append(x)
        ys.append(y)

        s = np.clip(S(x), -1.0 + 1e-12, 1.0 - 1e-12)
        V_local = tanh_profile(np.array([x]), V_left, V_right, d_m)[0]
        U_local = tanh_profile(np.array([x]), U_left, U_right, d_m)[0]
        n_local = float(n_x(V_local, U_local, gamma1=0.39, hvf2=6.39**2))
        B_local = B0 * np.tanh(x / d_m)
        dcdx = (
            2.0 * 1.054571817e-34
            * np.sqrt(np.pi * abs(n_local) * 1e15)
            / (1.602176634e-19 * max(abs(B_local), 1e-15))
        )
        curvature = -np.sign(B_local) / max(dcdx / 2.0, 1e-30)
        theta += curvature * ds_m
        x += np.cos(theta) * ds_m
        y += np.sin(theta) * ds_m

    return np.asarray(xs), np.asarray(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vt", type=float, default=0.0)
    parser.add_argument("--vb", type=float, default=None,
                        help="Jedno Vb; domyslnie liczone sa wszystkie Vb.")
    parser.add_argument("--d-nm", type=float, default=60.0)
    parser.add_argument("--x-limit-nm", type=float, default=200.0)
    parser.add_argument("--y-limit-nm", type=float, default=150.0)
    parser.add_argument("--out-dir", default="data_d=60/trajectories_r_to_l")
    parser.add_argument("--positive-b", action="store_true",
                        help="Użyj B0=+B zamiast konwencji abc.py B0=-B.")
    args = parser.parse_args()

    d_m = args.d_nm * 1e-9
    x_min_m, x_max_m = -args.x_limit_nm * 1e-9, args.x_limit_nm * 1e-9
    y_min_m, y_max_m = -args.y_limit_nm * 1e-9, args.y_limit_nm * 1e-9
    vb_values = [args.vb] if args.vb is not None else list(B_VALUES_BY_VB)
    cap = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap, phys)

    for vb in vb_values:
        vb_key = min(B_VALUES_BY_VB, key=lambda value: abs(value - vb))
        if abs(vb_key - vb) > 1e-9:
            raise ValueError(f"Brak listy B dla Vb={vb}. Dostepne: {list(B_VALUES_BY_VB)}")
        B_values = B_VALUES_BY_VB[vb_key]
        state = solver.solve(args.vt, vb)
        n_left, n_right = state.n2, state.n1
        V_left = float(state.Vg2)
        V_right = float(state.Vg1)
        U_left = float(state.U2)
        U_right = float(state.U1)
        n_left_m2 = float(n_left) * 1e15
        n_right_m2 = float(n_right) * 1e15

        out_dir = Path(args.out_dir) / f"Vb{vb_key:.0f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        records = []
        for B_abs in B_values:
            B0 = -B_abs
            start = time.perf_counter()
            x_left, x_right = calculate_turning_distances(
                V_left, V_right, U_left, U_right, B0, d_m
            )
            xs, ys = trace_trajectory(
                V_left, V_right, U_left, U_right, B0, d_m,
                x_min_m, x_max_m, y_min_m, y_max_m,
            )
            elapsed = time.perf_counter() - start
            tag = f"B{B_abs:.2f}".replace('.', 'p')
            np.savez(
                out_dir / f"trajectory_{tag}.npz",
                x_nm=xs * 1e9, y_nm=ys * 1e9,
                B_T=B_abs, B0_T=B0, Vt=args.vt, Vb=vb,
                nL=n_left, nR=n_right, d_nm=args.d_nm,
                x_left_nm=np.nan if x_left is None else x_left * 1e9,
                x_right_nm=np.nan if x_right is None else x_right * 1e9,
            )
            ax.plot(xs * 1e9, ys * 1e9, lw=1.4, label=f"B={B_abs:.2f} T")
            records.append((B_abs, len(xs), x_left, x_right, elapsed))
            print(f"B={B_abs:5.2f} T: points={len(xs):7d}, "
                  f"dLturn={x_left * 1e9 if x_left is not None else np.nan:.3f} nm, "
                  f"dRturn={x_right * 1e9 if x_right is not None else np.nan:.3f} nm, "
                  f"time={elapsed:.3f} s")

        ax.axvline(0.0, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlim(-args.x_limit_nm, args.x_limit_nm)
        ax.set_ylim(-args.y_limit_nm, args.y_limit_nm)
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"Trajektorie semiklasyczne: Vt={args.vt:.0f} V, Vb={vb_key:.0f} V")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectories_all.png", dpi=180)
        fig.savefig(out_dir / "trajectories_all.pdf")
        plt.close(fig)

        np.savez(out_dir / "trajectories_summary.npz",
                 B_T=np.array([r[0] for r in records]),
                 n_points=np.array([r[1] for r in records]),
                 nL=n_left, nR=n_right, Vt=args.vt, Vb=vb,
                 d_nm=args.d_nm)
        print(f"Zapisano wyniki w: {out_dir}")


if __name__ == "__main__":
    main()
