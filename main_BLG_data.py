"""
Etap 1: generowanie i buforowanie danych do `data_d=0_AB/`. Nie tworzy żadnych wykresów.
"""

from pathlib import Path
import kwant  # noqa: F401
import numpy as np
import gc
import ctypes
from datetime import datetime
import sys
from scipy.integrate import quad
from scipy.optimize import brentq
from main_BLG import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_npz(path: str, **arrays):
    ensure_data_dir()
    np.savez(path, **arrays)


def _log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _cleanup_memory():
    """Spróbuj oddać pamięć po ciężkich krokach (Kwant/NumPy)."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Globalne wymiary układu transportowego
# ---------------------------------------------------------------------------
L = 200.0 / l_scale  # długość układu [j.a. Kwant]
W = 150.0 / l_scale  # szerokość układu [j.a. Kwant]
d = 0.001  / l_scale  # szerokość przejścia złącza [j.a. Kwant]

def calculate_dL_dR(nL, nR, B0, d):
    """
    Oblicza efektywne odległości dL i dR:

        dL = odległość w osi y dla jednego łuku po lewej
        dR = odległość w osi y dla jednego łuku po prawej

    Wynik w metrach.

    Docelowa dokładność:
        0.01 nm

    Model:
        n(x) = (nL+nR)/2 + (nR-nL)/2*tanh(x/d)

        B(x) = B0*tanh(x/d)

        dc(x) = 2*hbar*sqrt(pi*n(x))/(e*|B(x)|)
    """

    hbar = 1.054571817e-34
    e = 1.602176634e-19

    # =========================================================
    # PROFILE
    # =========================================================

    def n(x):
        return (
            (nL + nR) / 2
            + (nR - nL) / 2 * np.tanh(x / d)
        )

    def B(x):
        return B0 * np.tanh(x / d)

    # Zero of n(x): at most one (tanh is monotone). Computed analytically
    # so quad can split the interval there (integrable 1/sqrt singularity).
    if nL != nR:
        _t = -(nL + nR) / (nR - nL)
        x_n_zero = d * np.arctanh(_t) if abs(_t) < 1.0 else None
    else:
        x_n_zero = None

    # =========================================================
    # KRZYWIZNA
    # =========================================================

    def kappa(x):

        nx = abs(n(x))
        Bx = abs(B(x))

        if Bx == 0:
            return 0.0
        if nx == 0:
            nx = 1e-100  # singularity at CNP; handled by quad(points=x_n_zero)

        return (
            e * Bx
            / (hbar * np.sqrt(np.pi * nx))
        )

    # =========================================================
    # CAŁKA KRZYWIZNY
    #
    # theta(x) = theta(0) + integral(kappa dx / cos(theta))
    #
    # Wygodniej wykorzystujemy:
    #
    # sin(theta) = integral(kappa dx)
    #
    # przy theta(0)=0
    # =========================================================

    def S(x):

        if x >= 0:
            pts = [x_n_zero] if x_n_zero is not None and 0 < x_n_zero < x else []
            return quad(
                kappa,
                0,
                x,
                points=pts,
                epsabs=1e-14,
                epsrel=1e-12,
                limit=1000
            )[0]

        else:
            pts = [x_n_zero] if x_n_zero is not None and x < x_n_zero < 0 else []
            return -quad(
                kappa,
                x,
                0,
                points=pts,
                epsabs=1e-14,
                epsrel=1e-12,
                limit=1000
            )[0]

    # =========================================================
    # PUNKT ZAWROTU
    #
    # cos(theta)=0
    #
    # czyli |sin(theta)| = 1
    # =========================================================

    def find_turning_point(side):

        # szukamy w zakresie 0 ... 5 um
        x_grid = np.linspace(
            1e-15,
            5e-6,
            5000
        ) * side

        previous_x = x_grid[0]
        previous_f = abs(S(previous_x)) - 1.0

        for x in x_grid[1:]:

            current_f = abs(S(x)) - 1.0

            if previous_f * current_f < 0:

                return brentq(
                    lambda xx: abs(S(xx)) - 1.0,
                    previous_x,
                    x,
                    xtol=1e-14,
                    rtol=1e-12
                )

            previous_x = x
            previous_f = current_f

        return None

    # =========================================================
    # PUNKTY ZAWROTU
    # =========================================================

    xL = find_turning_point(-1)
    xR = find_turning_point(+1)

    if xL is None:
        raise RuntimeError(
            "Nie znaleziono punktu zawrotu po lewej stronie."
        )

    if xR is None:
        raise RuntimeError(
            "Nie znaleziono punktu zawrotu po prawej stronie."
        )

    # =========================================================
    # dy/dx
    # =========================================================

    def dydx(x):

        s = S(x)

        # zabezpieczenie numeryczne
        s = np.clip(
            s,
            -1 + 1e-14,
            1 - 1e-14
        )

        c = np.sqrt(1 - s**2)

        return s / c

    # =========================================================
    # ODLEGŁOŚĆ y
    # =========================================================

    yL = quad(
        dydx,
        xL,
        0,
        epsabs=1e-13,
        epsrel=1e-11,
        limit=2000
    )[0]

    yR = quad(
        dydx,
        0,
        xR,
        epsabs=1e-13,
        epsrel=1e-11,
        limit=2000
    )[0]

    # pełny łuk:
    #
    # 0 -> punkt zawrotu -> 0
    #
    dL = 2 * abs(yL)
    dR = 2 * abs(yR)

    return dL, dR


def calculate_distance_map(Vt_target=0.0):
    print("\n" + "=" * 70)
    print("DISTANCE MAP: dL i dR w funkcji Vb i B")
    print("=" * 70)
    ensure_data_dir()

    

    d_m = 60e-9  # szerokość przejścia złącza [m]

    cache_in  = f"data_d=0_AB/zadanie8_line_n1_n2_Vt_{Vt_target}.npz"
    cache_out = f"data_d=0_AB/distance_map_Vt{Vt_target:.1f}_d_{d_m*1e9:.1f}.npz"

    if Path(cache_out).exists():
        print(f"Dane już istnieją: {cache_out} — pomijam.")
        return

    if not Path(cache_in).exists():
        raise FileNotFoundError(
            f"Brak pliku wejściowego: {cache_in}\n"
            "Najpierw uruchom wykres_n1_n2_od_Vb()."
        )

    data_in = np.load(cache_in)
    n1_arr  = data_in["n1"]    # 10^15 m^-2, prawa warstwa BLG
    n2_arr  = data_in["n2"]    # 10^15 m^-2, lewa warstwa BLG
    Vb_arr  = data_in["Vb"]    # wartości napięcia Vb [V]

    n_Vb  = len(Vb_arr)
    B_arr = np.linspace(0.01, 10.0, 1000)  # [T]
    n_B   = len(B_arr)

    dL_map = np.full((n_Vb, n_B), np.nan)
    dR_map = np.full((n_Vb, n_B), np.nan)

    import time, signal

    def _alarm(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, _alarm)
    n_errors = 0

    for i_vb, (n1_val, n2_val) in enumerate(zip(n1_arr, n2_arr)):
        nR_SI = n1_val * 1e15  # prawa warstwa BLG1 [m^-2]
        nL_SI = n2_val * 1e15  # lewa warstwa BLG2  [m^-2]

        t0_vb = time.perf_counter()
        for i_b, B in enumerate(B_arr):
            try:
                signal.alarm(1)
                dL, dR = calculate_dL_dR(nL_SI, nR_SI, B, d_m)
                signal.alarm(0)
                dL_map[i_vb, i_b] = dL
                dR_map[i_vb, i_b] = dR
            except (RuntimeError, TimeoutError):
                signal.alarm(0)
                n_errors += 1
        dt_vb = time.perf_counter() - t0_vb

        if (i_vb + 1) % 50 == 0 or i_vb == n_Vb - 1:
            print(
                f"  Vb [{i_vb + 1:4d}/{n_Vb}]"
                f"  Vb={Vb_arr[i_vb]:.1f} V"
                f"  czas: {dt_vb:.1f} s"
                f"  błędów łącznie: {n_errors}"
            )

    signal.signal(signal.SIGALRM, signal.SIG_DFL)

    np.savez(
        cache_out,
        dL_map=dL_map,
        dR_map=dR_map,
        Vb=Vb_arr,
        B=B_arr,
        Vt=np.float64(Vt_target),
    )
    print(f"\nZapisano: {cache_out}  (błędów łącznie: {n_errors})")


# ===========================================================================
# ZADANIE 1 — dyspersja (s_f=1 i s_f=4)
# ===========================================================================
def zadanie1_dyspersja():
    print("\n" + "=" * 70)
    print("ZADANIE 1 (DATA): Dyspersja BLG")
    print("=" * 70)
    ensure_data_dir()
    common = {'L': 100/l_scale, 'W': 50/l_scale, 't': T_INTRALAYER, 'gamma1': GAMMA1}

    p1 = BLGSystemParameters(**common, s_f=1.0, U1=0.0, name="BLG_sf1")
    blg1 = make_blg_system(p1)
    e1=calculate_dispersion(blg1)          # auto-saves data_d=0_AB/BLG_sf1_dispersion.npy
    np.save(f"data_d=0_AB/{blg1.params.name}_dispersion.npy", e1)

    p2 = BLGSystemParameters(**common, s_f=4.0, U1=0.0, name="BLG_sf4")
    blg2 = make_blg_system(p2)
    e2 = calculate_dispersion(blg2)          # auto-saves data_d=0_AB/BLG_sf4_dispersion.npy
    np.save(f"data_d=0_AB/{blg2.params.name}_dispersion.npy", e2)

    np.save("data_d=0_AB/zadanie1_k_arr.npy", k_arr)
    print("  Zapisano: data_d=0_AB/BLG_sf1_dispersion.npy, data_d=0_AB/BLG_sf4_dispersion.npy")


# ===========================================================================
# ZADANIE 2 — dyspersja z przerwą i polem B
# ===========================================================================
def zadanie2_dyspersja_z_przerwa():
    print("\n" + "=" * 70)
    print("ZADANIE 2 (DATA): Dyspersja BLG z przerwą")
    print("=" * 70)
    ensure_data_dir()

    params = BLGSystemParameters(
        L=L, W=W, V1=0.0/E_scale,
        B=1.5/T_scale, s_f=4.0, t=T_INTRALAYER, gamma1=GAMMA1,
        U1=0.1/E_scale, d=d, name="BLG_z_przerwa",
    )
    k_pts = k_arr / 4
    blg = make_blg_system(params)
    e3 = calculate_dispersion(blg, k_points=k_pts)   # auto-saves data_d=0_AB/BLG_z_przerwa_dispersion.npy
    np.save("data_d=0_AB/zadanie2_k_arr.npy", k_pts)
    np.save(f"data_d=0_AB/{blg.params.name}_dispersion.npy", e3)
    _save_npz("data_d=0_AB/zadanie2_params.npz",
              U1=np.array([params.U1 * E_scale]),
              B=np.array([params.B * T_scale]))
    print("  Zapisano: data_d=0_AB/BLG_z_przerwa_dispersion.npy")


# ===========================================================================
# ZADANIE 3 — przewodność vs pole B
# ===========================================================================
def zadanie3_przewodnosc_vs_B():
    print("\n" + "=" * 70)
    print("ZADANIE 3 (DATA): Przewodność vs B")
    print("=" * 70)
    ensure_data_dir()

    params = BLGSystemParameters(
        L=L, W=W, V1=0.1/E_scale, s_f=4.0,
        t=T_INTRALAYER, gamma1=GAMMA1, U1=0.1/E_scale, d=d,
        name="BLG_G_vs_B",
    )
    B_values = np.arange(0.0, 10.0, 0.01) / T_scale
    print(f"  Obliczanie G dla {len(B_values)} wartości B …")
    conductance_vs_magnetic_field(params, B_values)   # auto-saves
    print("  Zapisano: data_d=0_AB/BLG_G_vs_B_G_vs_B.npy, data_d=0_AB/BLG_G_vs_B_B_arr.npy")


# ===========================================================================
# ZADANIE 4 — parametry samouzgodnione (solver, brak Kwant)
# ===========================================================================
def zadanie4_parametry_samouzgodnione():
    print("\n" + "=" * 70)
    print("ZADANIE 4 (DATA): Parametry samouzgodnione")
    print("=" * 70)
    ensure_data_dir()
    cache = "data_d=0_AB/zadanie4_sc_params.npz"
    if Path(cache).exists():
        print("  Dane już istnieją."); return

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=0.0, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)
    Vb_arr = np.linspace(-60, 60, 601)
    res = solver.solve_1D_sweep(Vb_arr, Vt=0.0)
    _save_npz(cache, Vb=Vb_arr,
              n1=res['n1'], n2=res['n2'],
              dn1=res['dn1'], dn2=res['dn2'],
              U1=res['U1'], U2=res['U2'],
              Vg1=res['Vg1'], Vg2=res['Vg2'])
    print(f"  Zapisano: {cache}")


def calculate_blg_solver_parameters(
    Vt: float = 0.0,
    Vb_min: float = 0.0,
    Vb_max: float = 50.0,
    Vb_step: float = 0.02,
    output_file: str | None = None,
    overwrite: bool = False,
):
    """Policz wszystkie parametry ``DoubleBLGSolver`` w funkcji ``Vb``.

    Zakres domyślny zawiera 2501 punktów: ``Vb=0, 0.02, ..., 50 V``.
    Wynik jest zapisywany jako ``.npz`` i zawiera osobną tablicę dla każdego
    parametru stanu solvera: ``n1``, ``dn1``, ``U1``, ``Vg1``, ``n2``,
    ``dn2``, ``U2`` oraz ``Vg2``.

    ``n1``/``n2`` są w jednostkach używanych przez solver (10^15 m^-2),
    ``U1``/``U2`` w eV, a ``Vg1``/``Vg2`` w V.
    """
    print("\n" + "=" * 70)
    print("SWEEP PARAMETRÓW DoubleBLGSolver")
    print("=" * 70)
    ensure_data_dir()

    if Vb_step <= 0:
        raise ValueError("Vb_step musi być dodatni.")
    if Vb_max < Vb_min:
        raise ValueError("Vb_max musi być większe lub równe Vb_min.")

    n_points = int(round((Vb_max - Vb_min) / Vb_step)) + 1
    Vb_arr = Vb_min + Vb_step * np.arange(n_points, dtype=float)
    # Wymuś dokładną wartość końca zakresu mimo błędów reprezentacji float.
    if np.isclose(Vb_arr[-1], Vb_max, rtol=0.0, atol=1e-12):
        Vb_arr[-1] = Vb_max

    if output_file is None:
        output_file = (
            f"data_d=0_AB/BLG_solver_params_Vt{Vt:g}_Vb"
            f"{Vb_min:g}-{Vb_max:g}_step{Vb_step:g}.npz"
        )

    if Path(output_file).exists() and not overwrite:
        print(f"  Dane już istnieją: {output_file} — pomijam.")
        return dict(np.load(output_file, allow_pickle=False))

    cap_params = CapacitanceParameters(
        Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0
    )
    phys_params = BLGPhysicsParameters(
        gamma1=0.39, hvf2=6.39**2, n_t=118.57
    )
    solver = DoubleBLGSolver(cap_params, phys_params)

    print(
        f"  Vt={Vt:g} V, Vb={Vb_arr[0]:g}..{Vb_arr[-1]:g} V, "
        f"krok={Vb_step:g} V ({len(Vb_arr)} punktów)"
    )
    result = solver.solve_1D_sweep(Vb_arr, Vt=Vt, use_previous=True)

    _save_npz(
        output_file,
        Vt=np.array([Vt], dtype=float),
        Vb=Vb_arr,
        Vb_min=np.array([Vb_min], dtype=float),
        Vb_max=np.array([Vb_max], dtype=float),
        Vb_step=np.array([Vb_step], dtype=float),
        n1=result["n1"], dn1=result["dn1"], U1=result["U1"], Vg1=result["Vg1"],
        n2=result["n2"], dn2=result["dn2"], U2=result["U2"], Vg2=result["Vg2"],
    )
    print(f"  Zapisano: {output_file}")
    return result


# ===========================================================================
# ZADANIE 5 — przewodność z samouzgodnionymi parametrami
# ===========================================================================
def zadanie5_przewodnosc_samouzgodniona():
    print("\n" + "=" * 70)
    print("ZADANIE 5 (DATA): Przewodność samouzgodniona")
    print("=" * 70)
    ensure_data_dir()
    cache_G  = "data_d=0_AB/zadanie5_G.npy"
    cache_sc = "data_d=0_AB/zadanie5_sc_params.npz"
    if Path(cache_G).exists() and Path(cache_sc).exists():
        print("  Dane już istnieją."); return

    params = BLGSystemParameters(
        L=L, W=W, s_f=4.0,
        t=T_INTRALAYER, gamma1=GAMMA1, d=d, name="BLG_selfconsistent",
    )
    Vb_arr = np.linspace(-60, 60, 601)
    print(f"  Obliczanie G dla {len(Vb_arr)} wartości Vb …")
    G, sc = conductance_with_selfconsistent_params(params, Vb_arr, Vt=0.0)
    np.save(cache_G, G)
    np.save("data_d=0_AB/zadanie5_Vb_arr.npy", Vb_arr)
    _save_npz(cache_sc, n1=sc['n1'], n2=sc['n2'], U1=sc['U1'], U2=sc['U2'],
              Vg1=sc['Vg1'], Vg2=sc['Vg2'])
    print(f"  Zapisano: {cache_G}, {cache_sc}")


# ===========================================================================
# ZADANIE 6 — mapy 2D samouzgodnione
# ===========================================================================
def zadanie6_mapy_2D():
    print("\n" + "=" * 70)
    print("ZADANIE 6 (DATA): Mapy 2D samouzgodnione")
    print("=" * 70)
    ensure_data_dir()

    cm = CapacitanceParameters.calculate_capacitance(epsilon_r=3.3, d_nm=1)
    cap_params  = CapacitanceParameters(Cm=cm, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)
    Vtg_arr = np.arange(-6.0, 6.1, 0.5)
    Vbg_arr = np.arange(-6.0, 6.1, 0.5)
    cache_file = "data_d=0_AB/zadanie6_mapy_2D.npy"
    cache_meta = "data_d=0_AB/zadanie6_mapy_2D_meta.npz"

    try:
        meta = np.load(cache_meta)
        if np.allclose(meta['Vtg'], Vtg_arr) and np.allclose(meta['Vbg'], Vbg_arr):
            print("  Dane już istnieją."); return
    except Exception:
        pass

    print(f"  Obliczam {len(Vtg_arr)}×{len(Vbg_arr)} pkt …")
    res = solver.solve_sweep(Vtg_arr, Vbg_arr)
    np.save(cache_file, res)
    np.savez(cache_meta, Vtg=Vtg_arr, Vbg=Vbg_arr)
    print(f"  Zapisano: {cache_file}")


# ===========================================================================
# ZADANIE 8 — mapa G(Vb, B) dla Vt=0
# ===========================================================================
def zadanie8_mapa_G_Vb_B_Vt0(uniform_B: bool = False):
    print("\n" + "=" * 70)
    field_label = "jednorodne B" if uniform_B else "profil B(x)"
    print(f"ZADANIE 8 (DATA): Mapa G(Vb, B) dla Vt=0 — {field_label}")
    print("=" * 70)
    ensure_data_dir()
    B_START, B_END, B_STEP = 0.5, 0.7, 0.1
    Vb_arr   = np.arange(0.0, 50.0, 0.1) 
    B_phys   = np.arange(B_START, B_END + 0.1*B_STEP, B_STEP)
    Vt_value = 0.0
    n_B, n_Vb = len(B_phys), len(Vb_arr)
    cache_suffix = "_uniformB" if uniform_B else ""
    cache_file = f"data_d=0_AB/zadanie8_G_map_Vt_B={B_START}_{B_END}{cache_suffix}.npy"
    cache_meta = f"data_d=0_AB/zadanie8_G_map_meta_Vt_B={B_START}_{B_END}{cache_suffix}.npz"

    try:
        G_map = np.load(cache_file)
        meta  = np.load(cache_meta)
        if (G_map.shape == (n_B, n_Vb)
                and np.allclose(meta["B"], B_phys)
                and np.allclose(meta["Vb"], Vb_arr)
                and np.isclose(float(meta["Vt"]), Vt_value)):
            print("  Dane już istnieją."); return
    except Exception:
        pass

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    base_params = BLGSystemParameters(
        L=L, W=W, s_f=4.0, t=T_INTRALAYER,
        gamma1=GAMMA1, name="BLG_z8_Vt0", d=d,
    )
    print(f"  Obliczam {n_B}×{n_Vb} pkt …")
    G_map = conductance_map_G_B_Vb(
        base_params, B_phys, Vb_arr, Vt=Vt_value,
        cap_params=cap_params, phys_params=phys_params,
        uniform_B=uniform_B, verbose=True,
    )
    np.save(cache_file, G_map)
    np.savez(cache_meta, B=B_phys, Vb=Vb_arr, Vt=Vt_value,
             uniform_B=uniform_B)
    print(f"  Zapisano: {cache_file}")


# ===========================================================================
# WYKRES G(B) dla stałego Vb
# ===========================================================================
def wykres_G_od_B_stale_Vb(Vb_target, Vt_target=0.0):
    print("\n" + "=" * 70)
    print(f"WYKRES G(B) (DATA): Vb={Vb_target} V, Vt={Vt_target} V")
    print("=" * 70)
    ensure_data_dir()

    cache_file = f"data_d=0_AB/zadanie8_line_Vb_{Vb_target}_Vt_{Vt_target}.npz"
    try:
        data = np.load(cache_file)
        if len(data['B']) == 500:
            print("  Dane już istnieją."); return
    except Exception:
        pass

    B_phys  = np.linspace(0.0, 10.0, 500)
    Vb_arr  = np.array([Vb_target])
    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    base_params = BLGSystemParameters(
        L=L, W=W, s_f=4.0, t=T_INTRALAYER,
        gamma1=GAMMA1, name=f"BLG_z8_line_Vb{Vb_target}_Vt{Vt_target}",
        d=d,
    )
    Path("data_d=0_AB").mkdir(parents=True, exist_ok=True)
    G_map    = conductance_map_G_B_Vb(
        base_params, B_phys, Vb_arr, Vt=Vt_target,
        cap_params=cap_params, phys_params=phys_params, verbose=True,
    )
    G_values = G_map[:, 0]
    np.savez(cache_file, G=G_values, B=B_phys, Vb=Vb_target, Vt=Vt_target)
    print(f"  Zapisano: {cache_file}")


# ===========================================================================
# ZADANIE 9 — prąd lokalny
# ===========================================================================
def zadanie9_prad_lokalny(Vt: float = 5.0, Vb: float = 5.0, B_T: float = 3.0):
    print("\n" + "=" * 70)
    print(f"ZADANIE 9 (DATA): Vt={Vt}V  Vb={Vb}V  B={B_T}T")
    print("=" * 70)
    ensure_data_dir()
    Path("data_d=0_AB/zadanie9_R_to_L_Bconst").mkdir(parents=True, exist_ok=True)

    cache_J    = f"data_d=0_AB/zadanie9_R_to_L_Bconst/zadanie9_current_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.2f}_J.npy"
    cache_meta = f"data_d=0_AB/zadanie9_R_to_L_Bconst/zadanie9_current_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.2f}_meta.npz"
    cache_file = cache_meta  # zwracamy ścieżkę meta jako główną referencję
    if Path(cache_J).exists() and Path(cache_meta).exists():
        print("  Dane już istnieją."); return cache_file

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    sc = DoubleBLGSolver(cap_params, phys_params).solve(Vt, Vb)

    V1 = sc.Vg1 / E_scale
    U1 = -sc.U1 / E_scale
    V2 = sc.Vg2 / E_scale
    U2 = sc.U2  / E_scale
    params = BLGSystemParameters(
        L= L, W=W, s_f=4.0, t=T_INTRALAYER, gamma1=GAMMA1,
        B=B_T/T_scale, V1=V1, U1=U1,
        V2=V2, U2=U2,
        name=f"z9_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}", d=d,
    )
    blg    = make_blg_system(params, use_potential_profile=True)
    energy = 0.0
    print(f"  Obliczam prąd lokalny dla E={energy:.3f} eV …")
    J_mods = calculate_total_bond_current(blg, energy)
    if len(J_mods) == 0:
        print("  Brak propagujących modów."); return cache_file
    

    ensure_data_dir()
    np.save(cache_J, np.array(J_mods))  # shape (n_mods, N_bonds)
    print(f"  Zapisano J: {cache_J}")
    del J_mods
    _cleanup_memory()

    print(f"  Obliczam przewodność G dla E={energy:.3f} eV …")
    G = calculate_conductance(blg, energy)
    _save_npz(
        cache_meta,
        G=np.array([G]), energy=np.array([energy]),
        Vt=np.array([Vt]), Vb=np.array([Vb]), B_T=np.array([B_T]),
        Vg1=np.array([sc.Vg1]), Vg2=np.array([sc.Vg2]),
        U1=np.array([sc.U1]), U2=np.array([sc.U2]),
        n1=np.array([sc.n1]), n2=np.array([sc.n2]),
        L=np.array([params.L]), W=np.array([params.W]),
        s_f=np.array([params.s_f]), gamma1=np.array([params.gamma1]),
        d=np.array([params.d]),
        V1=np.array([params.V1]), U1_kwant=np.array([params.U1]),
        V2=np.array([params.V2]), U2_kwant=np.array([params.U2]),
    )
    print(f"  Zapisano meta: {cache_meta}")
    return cache_file


# ===========================================================================yłem ProcessPoolExecutor (osobne p
# WYKRES A — elektrostatyka 1D
# ===========================================================================
def zadanie_A_elektrostatyka_1D():
    print("\n" + "=" * 70)
    print("WYKRES A (DATA): Elektrostatyka 1D")
    print("=" * 70)
    ensure_data_dir()
    cache = "data_d=0_AB/wykresA_sc_params.npz"
    if Path(cache).exists():
        print("  Dane już istnieją."); return

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)
    Vb_arr = np.linspace(-60, 60, 241)
    res = solver.solve_1D_sweep(Vb_arr, Vt=0.0)
    _save_npz(cache, Vb=Vb_arr, n1=res['n1'], n2=res['n2'], U1=res['U1'], U2=res['U2'])
    print(f"  Zapisano: {cache}")


# ===========================================================================
# WYKRES B — mapa 2D n_tot
# ===========================================================================
def zadanie_B_mapa_2D_ntot():
    print("\n" + "=" * 70)
    print("WYKRES B (DATA): Mapa 2D n_tot")
    print("=" * 70)
    ensure_data_dir()

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)
    Vt_arr = np.arange(-60.0, 61.0, 0.1)
    Vb_arr = np.arange(-60.0, 61.0, 0.1)
    cache_file = "data_d=0_AB/wykresB_ntot.npy"
    cache_meta = "data_d=0_AB/wykresB_ntot_meta.npz"

    try:
        meta = np.load(cache_meta)
        if np.allclose(meta['Vt'], Vt_arr) and np.allclose(meta['Vb'], Vb_arr):
            print("  Dane już istnieją."); return
    except Exception:
        pass

    print(f"  Obliczam {len(Vt_arr)}×{len(Vb_arr)} pkt …")
    res = solver.solve_sweep(Vt_arr, Vb_arr)
    np.save(cache_file, res)
    np.savez(cache_meta, Vt=Vt_arr, Vb=Vb_arr)
    print(f"  Zapisano: {cache_file}")


# ===========================================================================
# WYKRES C — dyspersja z B=0T i B=4T
# ===========================================================================
def zadanie_C_pasma_z_B():
    print("\n" + "=" * 70)
    print("WYKRES C (DATA): Dyspersja B=0T i B=4T")
    print("=" * 70)
    ensure_data_dir()

    s_f    = 4.0
    _scale = s_f * A_GRAPHENE / l_scale
    k_BZ_dim = np.linspace(-np.pi, np.pi, 500)
    k_pts    = k_BZ_dim / _scale
    U_gap    = 0.1 / E_scale
    common_kw = dict(L=L, W=W, s_f=s_f,
                     t=T_INTRALAYER, gamma1=GAMMA1, U1=U_gap, V1=0.0, d=d)

    print("  B = 0 T …")
    e1 = calculate_dispersion(
        make_blg_system(BLGSystemParameters(**common_kw, B=0.0, name="wykresC_B0T")),
        k_points=k_pts,
    )
    np.save("data_d=0_AB/wykresC_B0T_dispersion.npy", e1)   # auto-saves data_d=0_AB/wykresC_B0T_dispersion.npy

    print("  B = 4 T …")
    e2 = calculate_dispersion(
        make_blg_system(BLGSystemParameters(**common_kw, B=4.0/T_scale, name="wykresC_B4T")),
        k_points=k_pts,
    )
    np.save("data_d=0_AB/wykresC_B4T_dispersion.npy", e2)   # auto-saves data_d=0_AB/wykresC_B4T_dispersion.npy

    np.save("data_d=0_AB/wykresC_k_BZ_dim.npy", k_BZ_dim)
    _save_npz("data_d=0_AB/wykresC_params.npz", U_gap=np.array([U_gap * E_scale]))
    print("  Zapisano: data_d=0_AB/wykresC_B0T_dispersion.npy, data_d=0_AB/wykresC_B4T_dispersion.npy")


# ===========================================================================
# WYKRES D — mapa G(Vb, B)
# ===========================================================================
def zadanie_D_mapa_G_Vb_B():
    print("\n" + "=" * 70)
    print("WYKRES D (DATA): Mapa G(Vb, B)")
    print("=" * 70)
    ensure_data_dir()

    Vt_fixed = 0.0
    Vb_arr   = np.arange(0, 60.0, 0.2)
    B_phys   = np.arange(0.0, 10.0, 0.1)
    n_B, n_Vb = len(B_phys), len(Vb_arr)
    cache_file = "data_d=0_AB/wykresD_G_map.npy"
    cache_meta = "data_d=0_AB/wykresD_G_map_meta.npz"

    try:
        G_map = np.load(cache_file)
        meta  = np.load(cache_meta)
        if (G_map.shape == (n_B, n_Vb)
                and np.allclose(meta['Vt'], [Vt_fixed])
                and np.allclose(meta['B'], B_phys)
                and np.allclose(meta['Vb'], Vb_arr)):
            print("  Dane już istnieją."); return
    except Exception:
        pass

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    base_params = BLGSystemParameters(
        L=L, W=W, s_f=4.0,
        t=T_INTRALAYER, gamma1=GAMMA1, d=d, name="wykresD",
    )
    # Oblicz n_tot do linii R_c i zapisz razem z mapą
    sc       = DoubleBLGSolver(cap_params, phys_params).solve_1D_sweep(Vb_arr, Vt=Vt_fixed)
    n_tot_SI = np.abs(sc['n1'] + sc['n2']) * 1e15

    print(f"  Obliczam {n_B}×{n_Vb} pkt …")
    G_map = conductance_map_G_B_Vb(
        base_params, B_phys, Vb_arr, Vt=Vt_fixed,
        cap_params=cap_params, phys_params=phys_params, verbose=True,
    )
    np.save(cache_file, G_map)
    np.savez(cache_meta, Vt=[Vt_fixed], B=B_phys, Vb=Vb_arr, n_tot_SI=n_tot_SI)
    print(f"  Zapisano: {cache_file}")


# ===========================================================================
# WYKRES E — profil potencjału (czysto matematyczny, brak Kwant)
# ===========================================================================
def zadanie_E_profil_potencjalu(Vt=5.0, Vb=5.0, B_T=3.0, L_nm=L, d_nm=d):
    print("\n" + "=" * 70)
    print(f"WYKRES E (DATA): Vt={Vt} V  Vb={Vb} V  B={B_T} T")
    print("=" * 70)
    ensure_data_dir()

    cache_file = f"data_d=0_AB/wykresE_profil_potencjalu_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}.npz"
    if Path(cache_file).exists():
        print("  Dane już istnieją."); return cache_file

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    sc = DoubleBLGSolver(cap_params, phys_params).solve(Vt, Vb)
    Vg1_eV = sc.Vg1; U1_eV = -sc.U1; Vg2_eV = sc.Vg2; U2_eV = sc.U2

    x_nm   = np.linspace(-L_nm, L_nm, 4000)
    tanh_x = np.tanh(x_nm / d_nm)
    B_x    = B_T * tanh_x
    V_x    = (Vg1_eV + Vg2_eV)/2 + (Vg1_eV - Vg2_eV)/2 * tanh_x
    U_x    = (U1_eV  + U2_eV)/2  + (U1_eV  - U2_eV)/2  * tanh_x
    low_layer = V_x + U_x/2; upp_layer = V_x - U_x/2

    _save_npz(
        cache_file,
        x_nm=x_nm, B_x=B_x, V_x=V_x, U_x=U_x,
        low_layer=low_layer, upp_layer=upp_layer,
        Vg1_eV=np.array([Vg1_eV]), U1_eV=np.array([U1_eV]),
        Vg2_eV=np.array([Vg2_eV]), U2_eV=np.array([U2_eV]),
        n1=np.array([sc.n1]), n2=np.array([sc.n2]),
        Vt=np.array([Vt]), Vb=np.array([Vb]), B_T=np.array([B_T]),
        L_nm=np.array([L_nm]), d_nm=np.array([d_nm]),
    )
    print(f"  Zapisano: {cache_file}")
    return cache_file


# ===========================================================================
# WYKRES F — dyspersja pełnego układu (parametry samouzgodnione, oba leady)
# ===========================================================================
def zadanie_F_dyspersja(Vt: float = 0.0, Vb: float = 30.0, B_T: float = 1.67):
    print("\n" + "=" * 70)
    print(f"WYKRES F (DATA): Dyspersja układu z Vt={Vt}V  Vb={Vb}V  B={B_T}T")
    print("=" * 70)
    ensure_data_dir()

    tag = f"Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.2f}"
    cache_left  = f"data_d=0_AB/wykresF_{tag}_left_dispersion.npy"
    cache_right = f"data_d=0_AB/wykresF_{tag}_right_dispersion.npy"
    cache_meta  = f"data_d=0_AB/wykresF_{tag}_meta.npz"
    if Path(cache_left).exists() and Path(cache_right).exists():
        print("  Dane już istnieją."); return

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    sc = DoubleBLGSolver(cap_params, phys_params).solve(Vt, Vb)

    params = BLGSystemParameters(
        L=L, W=W, s_f=4.0, t=T_INTRALAYER, gamma1=GAMMA1,
        B=B_T / T_scale,
        V1=sc.Vg1 / E_scale,  U1=-sc.U1 / E_scale,
        V2=sc.Vg2 / E_scale,  U2=sc.U2  / E_scale,
        d=d, name=f"wykresF_{tag}",
    )
    blg = make_blg_system(params, use_potential_profile=True)

    _scale   = params.s_f * A_GRAPHENE / l_scale
    k_BZ_dim = np.linspace(-np.pi, np.pi, 500)
    k_pts    = k_BZ_dim / _scale

    print("  Lead 0 (lewa strona: V2, U2, −B) …")
    E_left  = calculate_dispersion(blg, lead_idx=0, k_points=k_pts)
    print("  Lead 1 (prawa strona: V1, U1, +B) …")
    E_right = calculate_dispersion(blg, lead_idx=1, k_points=k_pts)

    np.save(cache_left,  E_left)
    np.save(cache_right, E_right)
    _save_npz(cache_meta,
              k_BZ_dim=k_BZ_dim,
              Vt=np.array([Vt]), Vb=np.array([Vb]), B_T=np.array([B_T]),
              Vg1=np.array([sc.Vg1]), U1=np.array([sc.U1]),
              Vg2=np.array([sc.Vg2]), U2=np.array([sc.U2]),
              n1=np.array([sc.n1]),   n2=np.array([sc.n2]))
    print(f"  Zapisano: {cache_left}, {cache_right}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("SYMULACJE TRANSPORTU PRZEZ DWUWARSTWOWY GRAFEN — ETAP DANYCH")
    print("=" * 70)
    ensure_data_dir()

    for fn, label in [
        (zadanie1_dyspersja,              "zadanie 1"),
        (zadanie2_dyspersja_z_przerwa,    "zadanie 2"),
        # (zadanie3_przewodnosc_vs_B,       "zadanie 3"),
        (zadanie4_parametry_samouzgodnione, "zadanie 4"),
        (zadanie5_przewodnosc_samouzgodniona, "zadanie 5"),
        (zadanie6_mapy_2D,                "zadanie 6"),
        # (lambda: wykres_G_od_B_stale_Vb(Vt=0.0, Vb=20.0),  "wykres G(B)- stalym Vb"),
        # (zadanie8_mapa_G_Vb_B_Vt0,       "zadanie 8"),
        # (lambda: calculate_blg_solver_parameters(Vt=0.0,Vb_min=0.0,Vb_max=50.0,Vb_step=0.02,overwrite=True), "BLG solver values"),
        # (lambda: calculate_distance_map(), "Distance map"), 
        # (lambda: zadanie_F_dyspersja(Vt=0.0, Vb=30.0, B_T=1.67),  "wykres F"),
        (zadanie_A_elektrostatyka_1D,     "wykres A"),
        (zadanie_B_mapa_2D_ntot,          "wykres B"),
        (zadanie_C_pasma_z_B,             "wykres C"),
        # (zadanie_D_mapa_G_Vb_B,           "wykres D"),
        # (lambda: zadanie_E_profil_potencjalu(Vt=0.0, Vb=20.0, B_T=6), "wykres E"),
        # (lambda: zadanie_E_profil_potencjalu(Vt=0.0, Vb=30.0, B_T=6), "wykres E"),
        # (lambda: zadanie_E_profil_potencjalu(Vt=0.0, Vb=45.0, B_T=6), "wykres E"),
    ]:
        _log(f"START {label}")
        try:
            fn()
        except Exception as e:
            print(f"Błąd w {label}: {e}")
        finally:
            _cleanup_memory()
            _log(f"END {label}")

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO ETAP DANYCH")
    print("=" * 70)


if __name__ == "__main__":
    main()
