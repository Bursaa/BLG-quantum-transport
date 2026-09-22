"""
Etap 2: rysowanie wykresów na podstawie danych z `data_d=60_AB/`. Nie liczy nowych danych.
Jeśli cache nie istnieje, pomija dany wykres z komunikatem.
"""

from pathlib import Path
import warnings
import kwant
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import argrelextrema

from main_BLG import *  # noqa: F401,F403


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_npz(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return np.load(path, allow_pickle=True)


def _s(arr):
    """Wyciągnij skalar z tablicy npz."""
    return float(np.atleast_1d(arr)[0])


# ===========================================================================
# ZADANIE 1 — dyspersja
# ===========================================================================
def zadanie1_dyspersja():
    print("\n" + "=" * 70)
    print("ZADANIE 1 (PLOT): Dyspersja BLG")
    print("=" * 70)
    ensure_plots_dir()

    f1 = "data_d=60_AB/BLG_sf1_dispersion.npy"
    f2 = "data_d=60_AB/BLG_sf4_dispersion.npy"
    if not Path(f1).exists() or not Path(f2).exists():
        print("  Brak cache — pomijam."); return

    E1 = np.load(f1); E2 = np.load(f2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_arr, E1 * E_scale, 'b-', linewidth=0.5)
    axes[0].set_xlabel(r"$k$ (1/nm)"); axes[0].set_ylabel(r"$E$ (eV)")
    axes[0].set_ylim(-0.3, 0.3); axes[0].set_title(r"BLG $s_f = 1$ (oryginalna siatka)")
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5); axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_arr, E2 * E_scale, 'r-', linewidth=0.5)
    axes[1].set_xlabel(r"$k$ (1/nm)"); axes[1].set_ylabel(r"$E$ (eV)")
    axes[1].set_ylim(-0.3, 0.3); axes[1].set_title(r"BLG $s_f = 4$ (przeskalowana siatka)")
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/zadanie1_dyspersja_BLG.pdf")
    plt.savefig("plots_d=60_AB/zadanie1_dyspersja_BLG.png", dpi=150)
    print("Zapisano: plots_d=60_AB/zadanie1_dyspersja_BLG.pdf")


# ===========================================================================
# ZADANIE 2 — dyspersja z przerwą
# ===========================================================================
def zadanie2_dyspersja_z_przerwa():
    print("\n" + "=" * 70)
    print("ZADANIE 2 (PLOT): Dyspersja BLG z przerwą")
    print("=" * 70)
    ensure_plots_dir()

    f = "data_d=60_AB/BLG_z_przerwa_dispersion.npy"
    if not Path(f).exists():
        print("  Brak cache — pomijam."); return

    E     = np.load(f)
    k_pts = np.load("data_d=60_AB/zadanie2_k_arr.npy") if Path("data_d=60_AB/zadanie2_k_arr.npy").exists() else k_arr / 16
    U_eV, B_T_val = 0.1, 1.5
    if Path("data_d=60_AB/zadanie2_params.npz").exists():
        p = np.load("data_d=60_AB/zadanie2_params.npz")
        U_eV = _s(p['U1']); B_T_val = _s(p['B'])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k_pts, E * E_scale, 'b-', linewidth=0.5)
    ax.set_xlabel(r"$k$ (1/nm)"); ax.set_ylabel(r"$E$ (eV)")
    ax.set_ylim(-0.3, 0.3)
    ax.set_title(f"BLG z U = {U_eV:.2f} eV, B = {B_T_val:.1f} T")
    ax.axhline(0,         color='gray', linestyle='--', alpha=0.5)
    ax.axhline( U_eV/2,   color='red',  linestyle=':',  alpha=0.7, label='+U/2')
    ax.axhline(-U_eV/2,   color='red',  linestyle=':',  alpha=0.7, label='-U/2')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/zadanie2_dyspersja_z_przerwa.pdf")
    plt.savefig("plots_d=60_AB/zadanie2_dyspersja_z_przerwa.png", dpi=150)
    print("Zapisano: plots_d=60_AB/zadanie2_dyspersja_z_przerwa.pdf")


# ===========================================================================
# ZADANIE 3 — przewodność vs B
# ===========================================================================
def zadanie3_przewodnosc_vs_B():
    print("\n" + "=" * 70)
    print("ZADANIE 3 (PLOT): Przewodność vs B")
    print("=" * 70)
    ensure_plots_dir()

    fG = "data_d=60_AB/BLG_G_vs_B_G_vs_B.npy"
    fB = "data_d=60_AB/BLG_G_vs_B_B_arr.npy"
    if not Path(fG).exists() or not Path(fB).exists():
        print("  Brak cache — pomijam."); return None, None

    G = np.load(fG); B_values = np.load(fB)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(B_values * T_scale, G, 'bo-', markersize=4)
    ax.set_xlabel(r"$B$ (T)"); ax.set_ylabel(r"$G$ ($2e^2/h$)")
    ax.set_title("Przewodność dwuwarstwowego grafenu vs pole magnetyczne")
    ax.grid(True, alpha=0.3)

    maxima = argrelextrema(G, np.greater)[0]; minima = argrelextrema(G, np.less)[0]
    if len(maxima) > 0:
        ax.scatter(B_values[maxima] * T_scale, G[maxima], color='green', s=100, zorder=5, label='Maksima')
    if len(minima) > 0:
        ax.scatter(B_values[minima] * T_scale, G[minima], color='red', s=100, zorder=5, label='Minima')
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/zadanie3_G_vs_B.pdf")
    plt.savefig("plots_d=60_AB/zadanie3_G_vs_B.png", dpi=150)
    print("Zapisano: plots_d=60_AB/zadanie3_G_vs_B.pdf")
    return G, B_values


# ===========================================================================
# ZADANIE 4 — parametry samouzgodnione
# ===========================================================================
def zadanie4_parametry_samouzgodnione():
    print("\n" + "=" * 70)
    print("ZADANIE 4 (PLOT): Parametry samouzgodnione")
    print("=" * 70)
    ensure_plots_dir()

    cache = "data_d=0_AB/zadanie4_sc_params.npz"
    if not Path(cache).exists():
        print("  Brak cache — pomijam."); return

    d = np.load(cache)
    Vb_arr  = d['Vb']
    results = {k: d[k] for k in ['n1', 'n2', 'dn1', 'dn2', 'U1', 'U2', 'Vg1', 'Vg2']}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(Vb_arr, results['n1'], 'b-', label=r'$n_1$ (górny BLG)')
    axes[0, 0].plot(Vb_arr, results['n2'], 'r--', label=r'$n_2$ (dolny BLG)')
    axes[0, 0].set_xlabel(r'$V_b$ (V)'); axes[0, 0].set_ylabel(r'$n$ ($10^{15}$ m$^{-2}$)')
    axes[0, 0].set_title('Gęstość nośników'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    axes[0, 1].plot(Vb_arr, results['U1'], 'b-', label=r'$U_1$')
    axes[0, 1].plot(Vb_arr, results['U2'], 'r--', label=r'$U_2$')
    axes[0, 1].set_xlabel(r'$V_b$ (V)'); axes[0, 1].set_ylabel(r'$U$ (eV)')
    axes[0, 1].set_title('Parametr asymetrii'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)

    axes[1, 0].plot(Vb_arr, results['Vg1'], 'b-', label=r'$V_{G1}$')
    axes[1, 0].plot(Vb_arr, results['Vg2'], 'r--', label=r'$V_{G2}$')
    axes[1, 0].set_xlabel(r'$V_b$ (V)'); axes[1, 0].set_ylabel(r'$V_G$ (V)')
    axes[1, 0].set_title('Potencjał bramkowy'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(Vb_arr, results['dn1'], 'b-', label=r'$\Delta n_1$')
    axes[1, 1].plot(Vb_arr, results['dn2'], 'r--', label=r'$\Delta n_2$')
    axes[1, 1].set_xlabel(r'$V_b$ (V)'); axes[1, 1].set_ylabel(r'$\Delta n$ ($10^{15}$ m$^{-2}$)')
    axes[1, 1].set_title('Asymetria gęstości'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots_d=0_AB/zadanie4_parametry_samouzgodnione.pdf")
    plt.savefig("plots_d=0_AB/zadanie4_parametry_samouzgodnione.png", dpi=150)
    print("Zapisano: plots_d=0_AB/zadanie4_parametry_samouzgodnione.pdf")
    return results


# ===========================================================================
# ZADANIE 5 — przewodność samouzgodniona
# ===========================================================================
def zadanie5_przewodnosc_samouzgodniona():
    print("\n" + "=" * 70)
    print("ZADANIE 5 (PLOT): Przewodność samouzgodniona")
    print("=" * 70)
    ensure_plots_dir()

    cache_G  = "data_d=0_AB/zadanie5_G.npy"
    cache_sc = "data_d=0_AB/zadanie5_sc_params.npz"
    if not Path(cache_G).exists() or not Path(cache_sc).exists():
        print("  Brak cache — pomijam."); return None, None

    G      = np.load(cache_G)
    Vb_arr = np.load("data_d=0_AB/zadanie5_Vb_arr.npy") if Path("data_d=0_AB/zadanie5_Vb_arr.npy").exists() else np.linspace(-30, 30, 31)
    sc_d   = np.load(cache_sc)
    sc_results = {k: sc_d[k] for k in ['n1', 'n2', 'U1', 'U2', 'Vg1', 'Vg2']}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(Vb_arr, G, 'ko-', markersize=4)
    axes[0, 0].set_xlabel(r'$V_b$ (V)'); axes[0, 0].set_ylabel(r'$G$ ($2e^2/h$)')
    axes[0, 0].set_title('Przewodność przez złącze'); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(Vb_arr, sc_results['U2'], 'r-o', markersize=3)
    axes[0, 1].set_xlabel(r'$V_b$ (V)'); axes[0, 1].set_ylabel(r'$U_2$ (eV)')
    axes[0, 1].set_title('Przerwa energetyczna dolnego bilayera'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(Vb_arr, sc_results['n2'], 'b-o', markersize=3)
    axes[1, 0].set_xlabel(r'$V_b$ (V)'); axes[1, 0].set_ylabel(r'$n_2$ ($10^{15}$ m$^{-2}$)')
    axes[1, 0].set_title('Gęstość nośników dolny bilayer'); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)

    sc_m = axes[1, 1].scatter(sc_results['U2'], G, c=Vb_arr, cmap='coolwarm')
    plt.colorbar(sc_m, ax=axes[1, 1], label=r'$V_b$ (V)')
    axes[1, 1].set_xlabel(r'$U_2$ (eV)'); axes[1, 1].set_ylabel(r'$G$ ($2e^2/h$)')
    axes[1, 1].set_title(r'Przewodność vs przerwa'); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots_d=0_AB/zadanie5_przewodnosc_samouzgodniona.pdf")
    plt.savefig("plots_d=0_AB/zadanie5_przewodnosc_samouzgodniona.png", dpi=150)
    print("Zapisano: plots_d=0_AB/zadanie5_przewodnosc_samouzgodniona.pdf")
    return G, sc_results


# ===========================================================================
# ZADANIE 6 — mapy 2D
# ===========================================================================
def zadanie6_mapy_2D():
    print("\n" + "=" * 70)
    print("ZADANIE 6 (PLOT): Mapy 2D samouzgodnione")
    print("=" * 70)
    ensure_plots_dir()

    cache_file = "data_d=0_AB/zadanie6_mapy_2D.npy"
    cache_meta = "data_d=0_AB/zadanie6_mapy_2D_meta.npz"
    if not Path(cache_file).exists():
        print("  Brak cache — pomijam."); return

    res = np.load(cache_file, allow_pickle=True).item()
    meta = np.load(cache_meta)
    Vtg_arr = meta['Vtg']; Vbg_arr = meta['Vbg']
    Vtg_2D, Vbg_2D = np.meshgrid(Vtg_arr, Vbg_arr)

    def _map(key): return res[key].T

    U2_map   = _map('U2'); U1_map  = _map('U1')
    n1_map   = _map('n1'); n2_map  = _map('n2')
    ntot_map = n1_map + n2_map
    boff_map = _map('Vg1') - _map('Vg2')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(r"Parametry samouzgodnione: "
                 r"$C_t = C_m = 2.55\times10^{15}$ m$^{-2}$V$^{-1}$", fontsize=11)

    datasets = [
        (axes[0, 0], U2_map,   r"$U_2$ (eV)",                    "RdBu_r", r"$U_2$ — parametr asymerii dolny BLG",   True),
        (axes[0, 1], ntot_map, r"$n_t + n_b$ ($10^{15}$ m$^{-2}$)", "RdBu_r", r"$n = n_t + n_b$ — całkowita gęstość", False),
        (axes[0, 2], n1_map,   r"$n_t$ ($10^{15}$ m$^{-2}$)",    "RdBu_r", r"$n_1$ — gęstość górny BLG",   False),
        (axes[1, 0], n2_map,   r"$n_b$ ($10^{15}$ m$^{-2}$)",    "RdBu_r", r"$n_2$ — gęstość dolny BLG",   False),
        (axes[1, 1], boff_map, r"$V_{G1} - V_{G2}$ (V)",          "PiYG",   r"Band offset $= V_{G1} - V_{G2}$",   False),
        (axes[1, 2], U1_map,   r"$U_1$ (eV)",                    "RdBu_r", r"$U_1$ — parametr asymetrii górny BLG",          True),
    ]
    for ax, data, cblabel, cmap_name, title, add_gap_contour in datasets:
        vmax = np.max(np.abs(data)); vmin = -vmax if vmax > 0 else -1e-9
        pcm = ax.pcolormesh(Vtg_2D, Vbg_2D, data, cmap=cmap_name, vmin=vmin, vmax=vmax, shading='auto')
        cb  = fig.colorbar(pcm, ax=ax, pad=0.02); cb.set_label(cblabel, fontsize=8)
        ax.set_xlabel(r"$V_{tg}$ (V)"); ax.set_ylabel(r"$V_{bg}$ (V)"); ax.set_title(title, fontsize=9)
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
        if add_gap_contour and vmax > 1e-6:
            cs = ax.contour(Vtg_2D, Vbg_2D, data, levels=[0], colors='k', linewidths=1.5)
            ax.clabel(cs, fmt=r"$U=0$", fontsize=7, inline=True)

    plt.tight_layout()
    plt.savefig("plots_d=0_AB/zadanie6_mapy_2D.pdf")
    plt.savefig("plots_d=0_AB/zadanie6_mapy_2D.png", dpi=150)
    print("Zapisano: plots_d=0_AB/zadanie6_mapy_2D.pdf")
    return res


# ===========================================================================
# ZADANIE 8 — mapa G(Vb, B)
# ===========================================================================
def zadanie8_mapa_G_Vb_B_Vt0():
    print("\n" + "=" * 70)
    print("ZADANIE 8 (PLOT): Mapa G(Vb, B) dla Vt=0")
    print("=" * 70)
    ensure_plots_dir()

    cache_file = "data_d=60_AB/zadanie8_G_map_Vt0_full.npy"
    cache_meta = "data_d=60_AB/zadanie8_G_map_meta_Vt0_full.npz"
    if not Path(cache_file).exists():
        print("  Brak cache — pomijam."); return

    G_map   = np.load(cache_file); meta = np.load(cache_meta)
    B_phys  = meta["B"]; Vb_arr = meta["Vb"]
    B_2D, Vb_2D = np.meshgrid(B_phys, Vb_arr)

    fig, ax = plt.subplots(figsize=(10, 8))
    pcm  = ax.pcolormesh(B_2D, Vb_2D, G_map.T, cmap="viridis", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax, pad=0.03)
    cbar.set_label(r"Przewodność $G$ ($2e^2/h$)", fontsize=11)
    ax.set_xlabel(r"Pole magnetyczne $B$ (T)", fontsize=11)
    ax.set_ylabel(r"Napięcie $V_b$ (V)", fontsize=11)
    ax.set_title(
        r"Przewodność $G(V_b, B)$ złącza BLG dla $V_t = 0$ V" "\n"
        r"(parametry samouzgodnione, $C_t = C_m = 2.55\times10^{15}$ m$^{-2}$V$^{-1}$)",
        fontsize=12, pad=15,
    )
    ax.set_xticks(np.arange(np.floor(B_phys.min()), np.ceil(B_phys.max()) + 0.01, 0.5))
    ax.set_xticks(np.arange(np.floor(B_phys.min()), np.ceil(B_phys.max()) + 0.01, 0.1), minor=True)
    ax.set_yticks(np.arange(np.floor(Vb_arr.min() / 5) * 5, np.ceil(Vb_arr.max() / 5) * 5 + 0.01, 5))
    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.grid(True, which="minor", axis="x", linestyle=":", alpha=0.2)
    plt.tight_layout()
    plt.savefig("plots_d=60_AB/zadanie8_G_Vb_B_Vt0_v2.pdf")
    plt.savefig("plots_d=60_AB/zadanie8_G_Vb_B_Vt0_v2.png", dpi=150)
    print("Zapisano: plots_d=60_AB/zadanie8_G_Vb_B_Vt0_v2.pdf")
    return G_map


def zadanie8_mapa_G_Vb_B_Vt0_z_poziomami_Landaua(
    n_max=15,
    landau_tolerance_frac=0.0,
    plot_difference_only=False,
    map_file="data_d=0_AB/zadanie8_G_map_Vt0_full_LEFT.npy",
    meta_file="data_d=0_AB/zadanie8_G_map_meta_Vt0_full_LEFT.npz",
    solver_file="data/BLG_solver_params_Vt0_Vb0-50_step0.02.npz",
):
    """Mapa ``G(Vb, B)`` z liniami poziomów Landaua.

    ``landau_tolerance_frac`` określa względną tolerancję dopasowania linii:
    przy wartości ``0.05`` zaznaczany jest obszar, w którym
    ``|E_n - (V_2 + U_2/2)| <= 5% |V_2 + U_2/2|``. Nominalna linia
    ``E_n = V_2 + U_2/2`` pozostaje narysowana pośrodku tego obszaru.

    Dla każdego ``n`` rysowany jest kontur równania

        E_n(B) = V2(Vb) + U2(Vb)/2,

    gdzie ``V2`` i ``U2`` pochodzą z samouzgodnionego ``DoubleBLGSolver``.
    Lewą stronę liczymy w eV, używając

        E_n = sign(n) * hbar * omega_c * sqrt(|n| * (|n| + 1)),
        omega_c = e B / m_eff.

    Domyślnie używana jest masa efektywna BLG
    ``m_eff = gamma1 / (2 v_F^2)``.
    """
    print("\n" + "=" * 70)
    print("WYKRES G(Vb, B) Z POZIOMAMI LANDAUA")
    print("=" * 70)
    if not 0.0 <= landau_tolerance_frac:
        raise ValueError("landau_tolerance_frac musi być nieujemne.")
    print(f"Tolerancja linii Landaua: {landau_tolerance_frac:.1%}")
    ensure_plots_dir()

    required = [Path(map_file), Path(meta_file), Path(solver_file)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("  Brak plików danych:")
        for path in missing:
            print(f"    - {path}")
        return None

    G_map = np.asarray(np.load(map_file), dtype=float)
    meta = np.load(meta_file)
    B_arr = np.asarray(meta["B"], dtype=float)
    Vb_arr = np.asarray(meta["Vb"], dtype=float)
    if G_map.shape != (len(B_arr), len(Vb_arr)):
        raise ValueError(
            f"Niepoprawny kształt mapy G: {G_map.shape}, "
            f"oczekiwano {(len(B_arr), len(Vb_arr))}."
        )

    solver_data = np.load(solver_file)
    solver_Vb = np.asarray(solver_data["Vb"], dtype=float)
    V1_solver = np.asarray(solver_data["Vg1"], dtype=float)
    V2_solver = np.asarray(solver_data["Vg2"], dtype=float)
    U1_solver = np.asarray(solver_data["U1"], dtype=float)
    U2_solver = np.asarray(solver_data["U2"], dtype=float)
    order = np.argsort(solver_Vb)
    solver_Vb = solver_Vb[order]
    V1_solver = V1_solver[order]
    V2_solver = V2_solver[order]
    U1_solver = U1_solver[order]
    U2_solver = U2_solver[order]
    if Vb_arr.min() < solver_Vb.min() or Vb_arr.max() > solver_Vb.max():
        raise ValueError(
            "Zakres Vb mapy wykracza poza zakres zapisanych parametrów solvera."
        )

    # hbar*e/m_eff przeliczone z J na eV/T.
    hbar_SI = 1.054571817e-34
    e_SI = 1.602176634e-19
    m_e = 9.1093837015e-31
    gamma1_J = 0.39 * e_SI
    v_F = 0.971 * 1.0e6
    m_eff = gamma1_J / (2.0 * v_F**2)

    # Energia odniesienia z prawej strony równania, w eV.
    landau_target_eV = e_SI*np.interp(Vb_arr, solver_Vb, V2_solver + U2_solver/2.0)
    hbar_omega_c_eV_per_T = hbar_SI * e_SI / m_eff 

    B_2D, Vb_2D = np.meshgrid(B_arr, Vb_arr)
    if plot_difference_only:
        n_limit = int(n_max)
        if n_limit < 6:
            raise ValueError("n_max musi być co najmniej równe 6 dla 12 paneli.")
        n_values = np.concatenate((
            -np.linspace(n_limit, 1, 6, dtype=int),
            np.linspace(1, n_limit, 6, dtype=int),
        ))
    else:
        n_values = range(0, int(n_max) + 1)

    if plot_difference_only:
        difference_fig, difference_axes = plt.subplots(
            3, 4, figsize=(16, 11), sharex=True, sharey=True,
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 8))
        pcm = ax.pcolormesh(
            B_2D, Vb_2D, G_map.T, cmap="viridis", shading="auto"
        )
        fig.colorbar(pcm, ax=ax, pad=0.03, label=r"$G$ ($2e^2/h$)")

    contour_handles = []
    for panel_index, n in enumerate(n_values):
        abs_n = abs(n)
        prefactor = hbar_omega_c_eV_per_T * np.sqrt(abs_n * (abs_n + 1))
        E_n = np.sign(n) * prefactor * B_2D
        # B_2D/Vb_2D mają kształt (n_Vb, n_B), więc cel zależny od Vb
        # trzeba rozszerzyć wzdłuż osi B: (n_Vb, 1).
        landau_target_2D = np.broadcast_to(
            landau_target_eV[:, None], E_n.shape
        )
        difference = E_n - landau_target_2D

        if plot_difference_only:
            difference_ax = difference_axes.flat[panel_index]
            difference_limit = max(float(np.nanmax(np.abs(difference))), 1e-12)
            difference_norm = TwoSlopeNorm(
                vmin=-difference_limit,
                vcenter=0.0,
                vmax=difference_limit,
            )
            difference_pcm = difference_ax.pcolormesh(
                B_2D, Vb_2D, difference,
                cmap="coolwarm", norm=difference_norm, shading="auto",
            )
            difference_fig.colorbar(
                difference_pcm, ax=difference_ax, pad=0.02,
                label=r"$E_n-(V_2)$ (eV)",
            )
            difference_ax.contour(
                B_2D, Vb_2D, difference,
                levels=[0.0], colors="black", linewidths=1.2,
            )
            difference_ax.set_title(rf"$n={n}$")
            difference_ax.set_xlabel(r"$B$ (T)")
            difference_ax.grid(True, linestyle="--", alpha=0.25)
            if panel_index % 4 == 0:
                difference_ax.set_ylabel(r"$V_b$ (V)")
            continue

        # Tolerancję liczymy względnie względem energii odniesienia. Na
        # wypadek wartości bliskich zeru używamy małego progu, aby nie
        # dopuścić do dzielenia przez zero ani do nieskończenie wąskiego pasa.
        reference_scale = np.maximum(np.abs(landau_target_eV), 1e-12)[:, None]
        relative_difference = difference / reference_scale
        if landau_tolerance_frac > 0:
            ax.contourf(
                B_2D, Vb_2D, relative_difference,
                levels=[-landau_tolerance_frac, landau_tolerance_frac],
                colors=["tomato"], alpha=0.08,
            )

        contours = ax.contour(
            B_2D, Vb_2D, difference,
            levels=[0.0], linewidths=1.5,
            colors=["tomato"],
        )
        if contours.allsegs[0]:
            contour_handles.append(
                Line2D(
                    [0], [0],
                    color="tomato",
                    lw=1.5,
                    label=rf"$n={n}$",
                )
            )

    if plot_difference_only:
        difference_fig.suptitle(
            r"Różnica $E_n(B)-(V_2(V_b) + U_2(V_b)/2)$",
            fontsize=12,
        )
        difference_fig.tight_layout()
        difference_base = "plots_d=60_AB/zadanie8_Landau_difference"
        difference_fig.savefig(difference_base + ".png", dpi=150)
        difference_fig.savefig(difference_base + ".pdf")
        plt.close(difference_fig)
        print(f"  Zapisano: {difference_base}.png/.pdf")
        return {
            "B": B_arr,
            "Vb": Vb_arr,
            "G": G_map,
            "landau_target_eV": landau_target_eV,
        }

    ax.set_xlabel(r"Pole magnetyczne $B$ (T)")
    ax.set_ylabel(r"Napięcie $V_b$ (V)")
    ax.set_xlim(B_arr.min(), B_arr.max())
    ax.set_ylim(Vb_arr.min(), Vb_arr.max())
    ax.set_title(
        r"Mapa $G(V_b,B)$ z liniami poziomów Landaua" "\n"
        r"$E_n=\operatorname{sgn}(n)\hbar\omega_c\sqrt{|n|(|n|+1)}= V_2 + U_2/2$",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.25)
    if contour_handles:
        ax.legend(handles=contour_handles, loc="upper right", framealpha=0.8)

    filename_base = f"plots_d=0_AB/zadanie8_G_map_Vt0_Landau_n{int(n_max)}_V2+U2_over_2_LEFT"
    fig.tight_layout()
    fig.savefig(filename_base + ".png", dpi=150)
    fig.savefig(filename_base + ".pdf")
    plt.close(fig)
    print(f"  Zapisano: {filename_base}.png/.pdf")
    print(f"  m_eff={m_eff / m_e:.4f} m_e, hbar*omega_c={hbar_omega_c_eV_per_T:.6f} eV/T")
    return {
        "B": B_arr,
        "Vb": Vb_arr,
        "G": G_map,
        "landau_target_eV": landau_target_eV,
        "m_eff": m_eff,
        "hbar_omega_c_eV_per_T": hbar_omega_c_eV_per_T,
    }


# ===========================================================================
# WYKRES G(B) dla stałego Vb
# ===========================================================================
def wykres_G_od_B_stale_Vb(Vb_target, Vt_target=0.0, width_nm=300.0, buffer_frac=0.05):
    print("\n" + "=" * 70)
    print(f"WYKRES G(B) (PLOT): Vb={Vb_target} V, Vt={Vt_target} V")
    print("=" * 70)
    ensure_plots_dir()

    cache_file = f"data_d=60_AB/zadanie8_line_Vb_{Vb_target}_Vt_{Vt_target}.npz"
    if not Path(cache_file).exists():
        print("  Brak cache — pomijam."); return None, None

    data     = np.load(cache_file)
    G_values = np.asarray(data['G'], dtype=float)
    B_phys   = np.asarray(data['B'], dtype=float)

    # Distance map ma wiersze dla kolejnych Vb i kolumny dla kolejnych B.
    # Punkty warunków są nanoszone na ten sam przekrój G(B).
    distance_candidates = [
        Path(f"data_d=60_AB/distance_map_Vt{Vt_target:.0f}.npz")
    ]
    distance_file = next((path for path in distance_candidates if path.exists()), None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(B_phys, G_values, '-', marker='o', markersize=4, color='teal',
            linewidth=1.5, label=r'$G(B)$')

    if distance_file is not None:
        distance_data = np.load(distance_file, allow_pickle=True)
        dL_map = np.asarray(distance_data['dL_map'], dtype=float)
        dR_map = np.asarray(distance_data['dR_map'], dtype=float)
        distance_Vb = np.asarray(distance_data['Vb'], dtype=float)
        distance_B = np.asarray(distance_data['B'], dtype=float)

        vb_idx = int(np.argmin(np.abs(distance_Vb - Vb_target)))
        dL = dL_map[vb_idx]
        dR = dR_map[vb_idx]
        S = dL + dR
        width_m = width_nm * 1e-9
        valid = np.isfinite(S) & np.isfinite(dR)
        if np.any(valid):
            # Mapa i przekrój mogą mieć różne siatki B, dlatego interpolujemy G
            # na pola z distance map zamiast zakładać identyczne indeksy.
            order = np.argsort(B_phys)
            G_at_distance_B = np.interp(
                distance_B, B_phys[order], G_values[order], left=np.nan, right=np.nan
            )
            min_positive = np.min(S[valid & (S > 0)]) if np.any(valid & (S > 0)) else 1e-12
            m_max = max(2, int(np.ceil(width_m / max(min_positive, 1e-12))) + 5)
            max_B, max_G, min_B, min_G = [], [], [], []

            for m in range(m_max + 1):
                target = width_m - dR / 2.0
                tol = np.maximum(1e-12, buffer_frac * np.abs(target))
                max_mask = valid & (np.abs(m * S - target) <= tol)

                min_target = width_m - dR / 2.0
                min_tol = np.maximum(1e-12, buffer_frac * np.abs(min_target))
                min_mask = valid & (np.abs(m * S + dR - min_target) <= min_tol)

                max_mask &= np.isfinite(G_at_distance_B)
                min_mask &= np.isfinite(G_at_distance_B)
                max_B.extend(distance_B[max_mask]); max_G.extend(G_at_distance_B[max_mask])
                min_B.extend(distance_B[min_mask]); min_G.extend(G_at_distance_B[min_mask])

            if max_B:
                ax.scatter(max_B, max_G, color='black', s=42, zorder=5,
                           label=r'maksima: $m(d_L+d_R)=W-d_R/2$')
            if min_B:
                ax.scatter(min_B, min_G, facecolors='white', edgecolors='black',
                           linewidths=0.9, s=48, zorder=6,
                           label=r'minima: $m(d_L+d_R)+d_R=W-d_R/2$')
        else:
            print("  Distance map nie zawiera poprawnych wartości dla tego Vb.")
    else:
        print("  Brak distance map — wykres zostanie bez punktów warunków.")

    ax.set_title(
        r"Przewodność $G(B)$ dla złącza BLG" "\n"
        rf"(ustalone $V_b = {Vb_target}$ V, $V_t = {Vt_target}$ V)",
        fontsize=12, pad=15,
    )
    ax.set_xlabel(r"Pole magnetyczne $B$ (T)", fontsize=11)
    ax.set_ylabel(r"Przewodność $G$ ($2e^2/h$)", fontsize=11)
    ax.set_xlim(0.0, 10.0)
    ax.set_xticks(np.arange(0.0, 10.01, 0.5))
    ax.set_xticks(np.arange(0.0, 10.01, 0.1), minor=True)
    ax.set_ylim(bottom=0.0, top=np.max(G_values) * 1.1)
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)
    ax.minorticks_on()
    ax.legend(loc='best'); fig.tight_layout()
    filename_base = f"plots_d=60_AB/zadanie8_G_B_Vb_{Vb_target}_Vt_{Vt_target}"
    fig.savefig(f"{filename_base}.png", dpi=150)
    fig.savefig(f"{filename_base}.pdf")
    plt.close(fig)
    print(f"Wykres zapisano jako: {filename_base}.png/.pdf")
    return B_phys, G_values


# ===========================================================================
# WYKRES G(B) z pełnej mapy zadanie8
# ===========================================================================
def przekroj_G_dla_stalego_B(B_target, map_file="data_d=0_AB/zadanie8_G_map_Vt0_full.npy",
                            meta_file="data_d=0_AB/zadanie8_G_map_meta_Vt0_full.npz"):
    """Zwraca przekrój G(Vb) dla zadanego pola B z pełnej mapy G(Vb, B).

    Z założenia mapa ma kształt (n_B, n_Vb), więc dla wybranego B
    wybieramy wiersz o najbliższym B i zwracamy kolumny G(B_target, Vb).
    """
    if not Path(map_file).exists() or not Path(meta_file).exists():
        raise FileNotFoundError(f"Brak mapy {map_file} lub meta {meta_file}.")

    G_map = np.load(map_file)
    meta = np.load(meta_file)
    B_phys = np.asarray(meta["B"], dtype=float)
    Vb_arr = np.asarray(meta["Vb"], dtype=float)
    Vt_val = float(np.atleast_1d(meta["Vt"])[0]) if "Vt" in meta else 0.0

    if G_map.shape[0] != len(B_phys) or G_map.shape[1] != len(Vb_arr):
        raise ValueError(
            f"Kształt mapy {G_map.shape} nie zgadza się z meta: "
            f"(len(B)={len(B_phys)}, len(Vb)={len(Vb_arr)})."
        )

    idx_B = int(np.argmin(np.abs(B_phys - B_target)))
    B_actual = B_phys[idx_B]
    G_slice = np.asarray(G_map[idx_B, :], dtype=float)
    return Vb_arr, G_slice, B_actual, Vt_val


def wykres_G_od_Vb_przy_stalym_B(B_target, map_file="data_d=0_AB/zadanie8_G_map_Vt0_full_LEFT.npy",
                                meta_file="data_d=0_AB/zadanie8_G_map_meta_Vt0_full_LEFT.npz",
                                save_dir="plots_d=0_AB"):
    """Rysuje G(Vb) dla stałego B na podstawie pełnej mapy G(Vb, B)."""
    ensure_plots_dir(save_dir)
    Vb_arr, G_slice, B_actual, Vt_val = przekroj_G_dla_stalego_B(B_target, map_file, meta_file)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Vb_arr, G_slice, "-o", lw=1.8, markersize=3, color="tab:blue")
    ax.set_xlabel(r"Napięcie $V_b$ (V)", fontsize=11)
    ax.set_ylabel(r"Przewodność $G$ ($2e^2/h$)", fontsize=11)
    ax.set_title(
        rf"Przekrój $G(V_b)$ dla $B = {B_actual:.3f}$ T, $V_t = {Vt_val:.0f}$ V",
        fontsize=11,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_base = Path(save_dir) / f"G_vs_Vb_B_{B_actual:.3f}T_Vt{Vt_val:.0f}"
    fig.savefig(str(out_base) + ".png", dpi=150)
    fig.savefig(str(out_base) + ".pdf")
    plt.close(fig)
    print(f"Zapisano: {out_base}.png / {out_base}.pdf")
    return Vb_arr, G_slice


def wykres_G_od_B_z_mapy(
    Vb_targets=None,
    map_file="data_d=0_AB/zadanie8_G_map_Vt0_full_LEFT.npy",
    meta_file="data_d=0_AB/zadanie8_G_map_meta_Vt0_full_LEFT.npz",
):
    if Vb_targets is None:
        Vb_targets = [20.0, 30.0, 45.0]
    print("\n" + "=" * 70)
    print("WYKRES G(B) z mapy (PLOT): Vb =", Vb_targets)
    print("=" * 70)
    ensure_plots_dir("plots")

    if not Path(map_file).exists() or not Path(meta_file).exists():
        print("  Brak pliku mapy — pomijam."); return

    G_map  = np.load(map_file)           # shape (n_B, n_Vb)
    meta   = np.load(meta_file)
    B_phys = meta["B"]
    Vb_arr = meta["Vb"]
    Vt_val = float(np.atleast_1d(meta["Vt"])[0]) if "Vt" in meta else 0.0

    for vb_target in Vb_targets:
        idx = int(np.argmin(np.abs(Vb_arr - vb_target)))
        vb_actual = Vb_arr[idx]
        G_slice = G_map[:, idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(B_phys, 2*G_slice, "-o", lw=1.5, markersize=3, color="teal")
        ax.set_xlabel(r"Pole magnetyczne $B$ (T)", fontsize=12)
        ax.set_ylabel(r"Przewodność $\frac{G}{2}$ ($e^2/h$)", fontsize=12)
        ax.set_title(
            rf"$G(B)$ dla $V_b = {vb_actual:.1f}$ V,  $V_t = {Vt_val:.0f}$ V" "\n"
            rf"(dane z mapy: {Path(map_file).name})",
            fontsize=11,
        )
        ax.set_xlim(B_phys.min(), B_phys.max())
        ax.set_ylim(bottom=0)
        y_max = G_slice.max() * 1.1
        ax.set_yticks(np.arange(0, y_max + 1, 5))
        ax.set_yticks(np.arange(0, y_max + 1, 1), minor=True)
        for level in np.arange(4, max(4, np.ceil(y_max)), 4):
            ax.axhline(level, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":", alpha=0.4)
        ax.minorticks_on()
        plt.tight_layout()

        fname = f"plots_d=0_AB/G_vs_B_z_mapy_Vt{Vt_val:.0f}_Vb{vb_actual:.1f}_LEFT"
        plt.savefig(fname + ".png", dpi=150)
        plt.savefig(fname + ".pdf")
        plt.close(fig)
        print(f"  Zapisano: {fname}.png")


def wykres_distance_map_z_warunkami(Vt_target=0.0, width_nm=300.0, buffer_frac=0.05):
    """Nakłada warunki maksymalnie/minimalne na mapę G(Vb, B) z dL_map i dR_map."""
    print("\n" + "=" * 70)
    print(f"WYKRES warunków z distance map (PLOT): Vt={Vt_target} V, buffer={buffer_frac:.0%}")
    print("=" * 70)
    ensure_plots_dir(); ensure_data_dir()

    candidates = sorted(Path("data").glob(f"distance_map_Vt{Vt_target:.0f}_d_60.npz"))
    if not candidates:
        print("  Brak pliku distance_map — pomijam."); return

    data = np.load(candidates[-1], allow_pickle=True)
    dL_map = np.asarray(data["dL_map"], dtype=float)
    dR_map = np.asarray(data["dR_map"], dtype=float)
    Vb_arr = np.asarray(data["Vb"], dtype=float)
    B_arr  = np.asarray(data["B"], dtype=float)
    if dL_map.shape != dR_map.shape:
        raise ValueError("dL_map i dR_map mają różne rozmiary.")

    W_m = width_nm * 1e-9
    S_map = dL_map + dR_map
    valid = np.isfinite(S_map) & np.isfinite(dR_map)
    S_positive = S_map[valid]
    if S_positive.size == 0:
        print("  Brak poprawnych danych w distance_map — pomijam."); return

    min_positive = float(np.min(S_positive[S_positive > 0])) if np.any(S_positive > 0) else 1e-12
    m_max = 3

    G_map = None
    G_file = Path("data_d=60_AB/zadanie8_G_map_Vt0_full.npy")
    G_meta = Path("data_d=60_AB/zadanie8_G_map_meta_Vt0_full.npz")
    if G_file.exists() and G_meta.exists():
        G_map = np.load(G_file)
        meta = np.load(G_meta)
        B_g = np.asarray(meta["B"], dtype=float)
        Vb_g = np.asarray(meta["Vb"], dtype=float)
        if G_map.shape == (len(B_g), len(Vb_g)):
            G_map_ok = True
        else:
            G_map_ok = False
    else:
        G_map_ok = False

    B_2D, Vb_2D = np.meshgrid(B_arr, Vb_arr)

    fig, ax = plt.subplots(figsize=(10, 7))
    if G_map_ok:
        B_2D_G, Vb_2D_G = np.meshgrid(B_g, Vb_g)
        pcm = ax.pcolormesh(B_2D_G, Vb_2D_G, G_map.T, cmap="viridis", shading="auto")
        fig.colorbar(pcm, ax=ax, pad=0.03, label=r"$G$ ($2e^2/h$)")
        ax.set_title(r"Mapa $G(V_b, B)$ z naniesionymi liniami warunków")
    else:
        ax.set_title(r"Mapa $dL+dR$ z naniesionymi liniami warunków")
        pcm = ax.pcolormesh(B_2D, Vb_2D, (dL_map + dR_map).T, cmap="gray", shading="auto")
        fig.colorbar(pcm, ax=ax, pad=0.03, label=r"$d_L + d_R$ (m)")

    ax.set_xlabel(r"Pole magnetyczne $B$ (T)")
    ax.set_ylabel(r"Napięcie $V_b$ (V)")
    ax.set_xlim(B_arr.min(), B_arr.max())
    ax.set_ylim(Vb_arr.min(), Vb_arr.max())
    ax.grid(True, linestyle="--", alpha=0.35)

    for m in range(0, m_max + 1):
        target = W_m - dR_map / 2.0
        tol = np.maximum(1e-12, buffer_frac * np.abs(target))

        max_mask = np.isfinite(S_map) & np.isfinite(dR_map)
        max_mask &= np.abs(m * S_map - target) <= tol
        if np.any(max_mask):
            rows, cols = np.where(max_mask)
            pts = np.column_stack((B_arr[cols], Vb_arr[rows]))
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], "k-", lw=1.5, alpha=0.95)

        min_mask = np.isfinite(S_map) & np.isfinite(dR_map)
        min_target = W_m - dR_map / 2.0
        min_tol = np.maximum(1e-12, buffer_frac * np.abs(min_target))
        min_mask &= np.abs(m * S_map + dR_map - min_target) <= min_tol
        if np.any(min_mask):
            rows, cols = np.where(min_mask)
            pts = np.column_stack((B_arr[cols], Vb_arr[rows]))
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], "w-", lw=2.0, alpha=0.9)

    ax.legend(
        [
            plt.Line2D([0], [0], color='k', lw=1.5),
            plt.Line2D([0], [0], color='w', lw=2.0),
        ],
        [r"maksima: $m(d_L+d_R)=W-d_R/2$", r"minima: $m(d_L+d_R)+d_R=W-d_R/2$"],
        loc='upper right', frameon=True,
    )

    fname = f"plots_d=60_AB/distance_map_conditions_Vt{Vt_target:.1f}_W{width_nm:.0f}nm"
    plt.tight_layout()
    plt.savefig(fname + ".png", dpi=150)
    plt.savefig(fname + ".pdf")
    plt.close(fig)
    print(f"  Zapisano: {fname}.png")
    return fig


# ===========================================================================
# ZADANIE 9 — prąd lokalny
# ===========================================================================

def _find_trajectory_file(Vb, B_T, root="data_d=60_AB/trajectories_l_to_r"):
    """Znajdź plik trajektorii najbliższy punktowi (Vb, B)."""
    directory = Path(root) / f"Vb{Vb:.0f}"
    candidates = sorted(directory.glob("trajectory_B*.npz"))
    if not candidates:
        return None

    best = None
    best_delta = np.inf
    for candidate in candidates:
        try:
            with np.load(candidate, allow_pickle=False) as data:
                candidate_b = float(np.asarray(data["B_T"]).reshape(-1)[0])
                candidate_vb = float(np.asarray(data["Vb"]).reshape(-1)[0])
        except (KeyError, OSError, ValueError):
            continue
        delta = abs(candidate_b - B_T) + abs(candidate_vb - Vb)
        if delta < best_delta:
            best, best_delta = candidate, delta

    if best is None or best_delta > 1e-6:
        return None
    return best


def _current_entry_y_nm(syst, J_row, side="left"):
    """Znajdź y wejścia, gdzie największy prąd płynie od strony lewej lub prawej."""
    J_row = np.asarray(J_row, dtype=float)
    edges = list(syst.graph)
    if len(edges) != len(J_row):
        return None

    positions = syst.sites
    x_mid = np.asarray([
        0.5 * (positions[a].pos[0] + positions[b].pos[0])
        for a, b in edges
    ])
    y_mid = np.asarray([
        0.5 * (positions[a].pos[1] + positions[b].pos[1])
        for a, b in edges
    ])
    dx = np.asarray([
        positions[b].pos[0] - positions[a].pos[0]
        for a, b in edges
    ])
    finite_x = x_mid[np.isfinite(x_mid)]
    if finite_x.size == 0:
        return None

    if side == "left":
        edge = np.min(finite_x)
        x_cutoff = -106.0 / l_scale
        entry_mask = (x_mid >= edge) & (x_mid <= x_cutoff) & (np.abs(dx) > 0.0)
        side_label = "lewej"
    elif side == "right":
        edge = np.max(finite_x)
        x_cutoff = 106.0 / l_scale
        entry_mask = (x_mid <= edge) & (x_mid >= x_cutoff) & (np.abs(dx) > 0.0)
        side_label = "prawej"
    else:
        raise ValueError(f"Nieznany side={side!r}; oczekiwane 'left' lub 'right'.")

    if not np.any(entry_mask):
        return None

    # Nie wybieramy pojedynczego wiązania: na krawędzi grafu występują
    # lokalne maksima numeryczne. Sumujemy prąd w pasmach y i wybieramy
    # najsilniejszy kanał wejściowy.
    y_nm = y_mid[entry_mask] * l_scale
    current_in = np.maximum(J_row[entry_mask], 0.0)
    bin_width_nm = 8.0
    y_min = np.floor(np.min(y_nm) / bin_width_nm) * bin_width_nm
    y_max = np.ceil(np.max(y_nm) / bin_width_nm) * bin_width_nm
    bins = np.arange(y_min, y_max + bin_width_nm, bin_width_nm)
    if bins.size < 2:
        return float(np.average(y_nm, weights=current_in + 1e-30))

    bin_index = np.clip(np.digitize(y_nm, bins) - 1, 0, bins.size - 2)
    channel_strength = np.bincount(
        bin_index, weights=current_in, minlength=bins.size - 1
    )
    strongest = int(np.argmax(channel_strength))
    channel_mask = bin_index == strongest
    if not np.any(channel_mask) or channel_strength[strongest] <= 0.0:
        return None
    return float(np.average(y_nm[channel_mask],
                            weights=current_in[channel_mask] + 1e-30))


def _overlay_trajectory(ax, Vb, B_T, y_start_nm=None,
                        root="data_d=60_AB/trajectories_l_to_r"):
    """Nałóż trajektorię z początkiem w (x=0, y=y_start_nm)."""
    trajectory_file = _find_trajectory_file(Vb, B_T, root=root)
    if trajectory_file is None:
        print(f"    Brak trajektorii dla Vb={Vb:g}, B={B_T:g} T")
        return False

    with np.load(trajectory_file, allow_pickle=False) as data:
        if "x_left_nm_path" in data and "x_right_nm_path" in data:
            paths = [
                (data["x_left_nm_path"], data["y_left_nm_path"], "royalblue"),
                (data["x_right_nm_path"], data["y_right_nm_path"], "firebrick"),
            ]
        else:
            paths = [(data["x_nm"], data["y_nm"], "darkorange")]

    for path_index, (x_nm, y_nm, color) in enumerate(paths):
        x_nm = np.asarray(x_nm, dtype=float)
        y_nm = np.asarray(y_nm, dtype=float)
        if y_start_nm is not None and x_nm.size:
            # Każda gałąź z danych zaczyna się przy x≈±0.1 nm. Przesuwamy
            # ją tak, aby dokładnie zaczynała się na granicy x=0 i na y
            # wyznaczonym z największego prądu przy lewej krawędzi.
            x_nm = x_nm - x_nm[0]
            y_nm = y_nm - y_nm[0] + y_start_nm
        ax.plot(
            x_nm / l_scale,
            y_nm / l_scale,
            color=color, linewidth=2.0, zorder=8,
            label=("trajektoria semiklasyczna — lewo" if path_index == 0 and color != "darkorange"
                   else "trajektoria semiklasyczna" if color == "darkorange" else None),
        )

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value * l_scale:.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value * l_scale:.0f}"))
    return True


def _plot_single_current(syst, J_row, title, fname_base, Vb=None, B_T=None):
    fig, ax = plt.subplots(figsize=(13, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kwant.plotter.current(syst, J_row, ax=ax, colorbar=True, show=False, cmap='RdBu_r')
    if Vb is not None and B_T is not None:
        y_entry_nm = _current_entry_y_nm(syst, J_row, side="left")
        if y_entry_nm is not None:
            y_entry_nm += 5.0
            print(f"    Trajektoria start: x=0 nm, y={y_entry_nm:.2f} nm")
        _overlay_trajectory(ax, Vb, B_T, y_start_nm=y_entry_nm)
    ax.set_title(title)
    ax.set_xlabel(r"$x$ (nm)"); ax.set_ylabel(r"$y$ (nm)")
    ax.axvline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.85)
    plt.tight_layout()
    plt.savefig(fname_base + ".pdf"); plt.savefig(fname_base + ".png", dpi=150)
    plt.close(fig)
    print(f"  Zapisano: {fname_base}.png")


def zadanie9_prad_lokalny():
    """Generuje wykresy dla wszystkich plików zadanie9 *_J.npy w data_d=60_AB/."""
    print("\n" + "=" * 70)
    print("ZADANIE 9 (PLOT): Prąd lokalny — wszystkie dane")
    print("=" * 70)
    ensure_plots_dir()

    j_files = sorted(Path("data_d=60_AB/zadanie9_L_to_R").glob("zadanie9_current_*_J.npy"))
    if not j_files:
        print("  Brak plików _J.npy — pomijam."); return

    for j_file in j_files:
        meta_file = Path(str(j_file).replace("_J.npy", "_meta.npz"))
        if not meta_file.exists():
            print(f"  Brak meta dla {j_file.name} — pomijam."); continue

        print(f"\n  Przetwarzam: {j_file.name}")
        J_all = np.load(j_file)   # shape (n_mods, N_bonds)
        meta  = np.load(meta_file)

        Vt  = float(meta["Vt"][0]);  Vb  = float(meta["Vb"][0])
        B_T = float(meta["B_T"][0]); G   = float(meta["G"][0])
        n_mods = J_all.shape[0]
        print(f"    {n_mods} modów,  G = {G:.2f} (2e²/h)")
        if Vb  != 20.0 or B_T > 0.5:
            print(f"    Pomijam: Vb={Vb}, B_T={B_T}")
            continue
        params = BLGSystemParameters(
            L=float(meta["L"][0]),   W=float(meta["W"][0]),
            s_f=float(meta["s_f"][0]), t=T_INTRALAYER,
            gamma1=float(meta["gamma1"][0]),
            B=B_T / T_scale,
            V1=float(meta["V1"][0]),       U1=float(meta["U1_kwant"][0]),
            V2=float(meta["V2"][0]),       U2=float(meta["U2_kwant"][0]),
            d=float(meta["d"][0]),
            name=f"z9_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.2f}",
        )
        blg = make_blg_system(params, use_potential_profile=True)

        out_dir = Path(f"plots_d=60_AB/zadanie9_L_to_R/Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.2f}")
        out_dir.mkdir(parents=True, exist_ok=True)

        header = rf"$V_t={Vt:.0f}$ V,  $V_b={Vb:.0f}$ V,  $B={B_T:.2f}$ T,  $G={G:.2f}\,(2e^2/h)$"

        n_show = n_mods
        for i in range(n_show):
            if i < n_mods - 2:
                continue
            _plot_single_current(
                blg.system, J_all[i],
                title=rf"Prąd lokalny — mod {i+1}/{n_mods}  |  {header}",
                fname_base=str(out_dir / f"mod{i+1:03d}"),
                Vb=Vb, B_T=B_T,
            )

        J_sum = J_all.sum(axis=0)
        _plot_single_current(
            blg.system, J_sum,
            title=rf"Prąd lokalny — suma ({n_mods} modów)  |  {header}",
            fname_base=str(out_dir / "suma"),
            Vb=Vb, B_T=B_T,
        )


# ===========================================================================
# WYKRES A — elektrostatyka 1D
# ===========================================================================
def zadanie_A_elektrostatyka_1D():
    print("\n" + "=" * 70)
    print("WYKRES A (PLOT): Elektrostatyka 1D")
    print("=" * 70)
    ensure_plots_dir()

    cache = "data_d=60_AB/wykresA_sc_params.npz"
    if not Path(cache).exists():
        print("  Brak cache — pomijam."); return None

    d      = np.load(cache)
    Vb_arr = d['Vb']
    res    = {'n1': d['n1'], 'n2': d['n2'], 'U1': d['U1'], 'U2': d['U2']}

    fig, ax1 = plt.subplots(figsize=(9, 5))
    lns1 = ax1.plot(Vb_arr, res['n1'], 'b-', lw=1.8, label=r'$n_1$ (górny BLG)')
    lns2 = ax1.plot(Vb_arr, res['n2'], 'r--', lw=1.8, label=r'$n_2$ (dolny BLG)')
    ax1.set_xlabel(r'$V_b$ (V)', fontsize=12)
    ax1.set_ylabel(r'Gęstość $n$  ($10^{15}$ m$^{-2}$)', fontsize=12)
    ax1.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
    ax1.axvline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
    ax1.grid(True, alpha=0.25); ax1.set_xlim(-60, 60)

    ax2  = ax1.twinx()
    lns3 = ax2.plot(Vb_arr, res['U1'], 'b:', lw=1.8, label=r'$U_1$ (górny BLG)')
    lns4 = ax2.plot(Vb_arr, res['U2'], 'r-.', lw=1.8, label=r'$U_2$ (dolny BLG)')
    ax2.set_ylabel(r'Przerwa energetyczna $U$ (eV)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.axhline(0, color='gray', lw=0.3, ls='--', alpha=0.4)

    lns = lns1 + lns2 + lns3 + lns4
    ax1.legend(lns, [l.get_label() for l in lns], loc='upper left', fontsize=9)
    ax1.set_title(
        r'Wykres A — Elektrostatyka 1D podwójnego BLG' '\n'
        r'$V_t = 0$ V,  $C_t = C_m = 2.55\times10^{15}$ m$^{-2}$V$^{-1}$',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/wykresA_elektrostatyka_1D.pdf")
    plt.savefig("plots_d=60_AB/wykresA_elektrostatyka_1D.png", dpi=150)
    print("Zapisano: plots_d=60_AB/wykresA_elektrostatyka_1D.pdf")
    return res


# ===========================================================================
# WYKRES B — mapa 2D n_tot
# ===========================================================================
def zadanie_B_mapa_2D_ntot():
    print("\n" + "=" * 70)
    print("WYKRES B (PLOT): Mapa 2D n_tot")
    print("=" * 70)
    ensure_plots_dir()

    cache_file = "data_d=60_AB/wykresB_ntot.npy"
    cache_meta = "data_d=60_AB/wykresB_ntot_meta.npz"
    if not Path(cache_file).exists():
        print("  Brak cache — pomijam."); return None

    res      = np.load(cache_file, allow_pickle=True).item()
    meta     = np.load(cache_meta)
    Vt_arr   = meta['Vt']; Vb_arr = meta['Vb']
    Vt_2D, Vb_2D = np.meshgrid(Vt_arr, Vb_arr)
    n_tot_map = (res['n1'] + res['n2']).T
    U1_map    = res['U1'].T; U2_map = res['U2'].T

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(float(np.max(np.abs(n_tot_map))), 1e-9)
    pcm  = ax.pcolormesh(Vt_2D, Vb_2D, n_tot_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    cb   = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r'$n_{tot} = n_1 + n_2$  ($10^{15}$ m$^{-2}$)', fontsize=10)
    cs1 = ax.contour(Vt_2D, Vb_2D, U1_map, levels=[0], colors='black',    linewidths=2.0, linestyles='-')
    cs2 = ax.contour(Vt_2D, Vb_2D, U2_map, levels=[0], colors='limegreen', linewidths=2.0, linestyles='-')
    ax.clabel(cs1, fmt=r'$U_1=0$', fontsize=8, inline=True)
    ax.clabel(cs2, fmt=r'$U_2=0$', fontsize=8, inline=True)
    ax.set_xlabel(r'$V_t$ (V)', fontsize=12); ax.set_ylabel(r'$V_b$ (V)', fontsize=12)
    ax.set_title(
        r'Wykres B — Całkowita gęstość $n_{tot}(V_t, V_b)$' '\n'
        r'kontury: $U_1=0$ (czarny), $U_2=0$ (zielony)',
        fontsize=10,
    )
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax.legend(
        handles=[Line2D([0], [0], color='k', lw=2, label=r'$U_1 = 0$'),
                 Line2D([0], [0], color='limegreen', lw=2, label=r'$U_2 = 0$')],
        loc='lower right', fontsize=9,
    )

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/wykresB_mapa_2D_ntot.pdf")
    plt.savefig("plots_d=60_AB/wykresB_mapa_2D_ntot.png", dpi=150)
    print("Zapisano: plots_d=60_AB/wykresB_mapa_2D_ntot.pdf")
    return res


# ===========================================================================
# WYKRES C — dyspersja z B=0T i B=4T
# ===========================================================================
def zadanie_C_pasma_z_B():
    print("\n" + "=" * 70)
    print("WYKRES C (PLOT): Dyspersja B=0T i B=4T")
    print("=" * 70)
    ensure_plots_dir()

    f_B0 = "data_d=60_AB/wykresC_B0T_dispersion.npy"
    f_B4 = "data_d=60_AB/wykresC_B4T_dispersion.npy"
    if not Path(f_B0).exists() or not Path(f_B4).exists():
        print("  Brak cache — pomijam."); return

    E_B0     = np.load(f_B0); E_B4 = np.load(f_B4)
    k_BZ_dim = (np.load("data_d=60_AB/wykresC_k_BZ_dim.npy")
                if Path("data_d=60_AB/wykresC_k_BZ_dim.npy").exists()
                else np.linspace(-np.pi, np.pi, 500))
    U_gap_eV = (float(np.load("data_d=60_AB/wykresC_params.npz")['U_gap'])
                if Path("data_d=60_AB/wykresC_params.npz").exists() else 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(r'Wykres C — Struktura pasmowa BLG  ($s_f=4$, $W=50.0$ nm, $U=0.1$ eV)', fontsize=11)
    E_lim = 0.5

    for ax, E, col, title in [
        (ax1, E_B0, 'b', r'$B = 0$ T — gładkie pasma z przerwą'),
        (ax2, E_B4, 'r', r'$B = 4$ T — poziomy Landaua + stany krawędziowe'),
    ]:
        ax.plot(k_BZ_dim, E * E_scale, f'{col}-', lw=0.4, alpha=0.7)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-E_lim, E_lim)
        ax.set_xlabel(r'$k_y\, a$', fontsize=12); ax.set_title(title, fontsize=10)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.axhline(0,          color='gray',  lw=0.5, ls='--', alpha=0.5)
        ax.axhline( U_gap_eV/2, color='tomato', lw=0.8, ls=':', alpha=0.8, label=r'$\pm U/2$')
        ax.axhline(-U_gap_eV/2, color='tomato', lw=0.8, ls=':', alpha=0.8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax1.set_ylabel(r'$E$ (eV)', fontsize=12)

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/wykresC_pasma_z_B.pdf")
    plt.savefig("plots_d=60_AB/wykresC_pasma_z_B.png", dpi=150)
    print("Zapisano: plots_d=60_AB/wykresC_pasma_z_B.pdf")


# ===========================================================================
# WYKRES F — dyspersja pełnego układu (oba leady, parametry samouzgodnione)
# ===========================================================================
def zadanie_F_dyspersja():
    """Generuje wykresy dyspersji dla wszystkich plików wykresF_*_meta.npz w data_d=60_AB/."""
    print("\n" + "=" * 70)
    print("WYKRES F (PLOT): Dyspersja układu — wszystkie dane")
    print("=" * 70)
    ensure_plots_dir()

    meta_files = sorted(Path("data").glob("wykresF_*_meta.npz"))
    if not meta_files:
        print("  Brak plików wykresF_*_meta.npz — pomijam."); return

    for meta_file in meta_files:
        tag = meta_file.name.replace("wykresF_", "").replace("_meta.npz", "")
        f_left  = Path(f"data_d=60_AB/wykresF_{tag}_left_dispersion.npy")
        f_right = Path(f"data_d=60_AB/wykresF_{tag}_right_dispersion.npy")
        if not f_left.exists() or not f_right.exists():
            print(f"  Brak dyspersji dla {tag} — pomijam."); continue

        meta    = np.load(meta_file)
        E_left  = np.load(f_left)
        E_right = np.load(f_right)

        k_BZ_dim = meta["k_BZ_dim"]
        Vt  = float(meta["Vt"][0]); Vb  = float(meta["Vb"][0]); B_T = float(meta["B_T"][0])
        Vg1 = float(meta["Vg1"][0]); U1  = float(meta["U1"][0])
        Vg2 = float(meta["Vg2"][0]); U2  = float(meta["U2"][0])

        print(f"  {tag}  (Vt={Vt:.0f} V, Vb={Vb:.0f} V, B={B_T:.2f} T)")

        E_lim = max(0.3, 0.5 * max(abs(Vg1), abs(Vg2)) + 0.1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(
            rf"Wykres F — Dyspersja złącza BLG  "
            rf"$V_t={Vt:.0f}$ V,  $V_b={Vb:.0f}$ V,  $B={B_T:.2f}$ T",
            fontsize=11,
        )

        for ax, E, U_eV, V_eV, B_sign, title in [
            (ax1, E_left,  U2, Vg2, -B_T, rf"Lewa strona  ($V_{{g2}}={Vg2:.3f}$ eV,  $U_2={U2:.3f}$ eV,  $B={-B_T:.2f}$ T)"),
            (ax2, E_right, U1, Vg1, +B_T, rf"Prawa strona  ($V_{{g1}}={Vg1:.3f}$ eV,  $U_1={U1:.3f}$ eV,  $B={+B_T:.2f}$ T)"),
        ]:
            ax.plot(k_BZ_dim, E * E_scale, 'b-', lw=0.4, alpha=0.7)
            ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-E_lim, E_lim)
            ax.set_xlabel(r"$k_y\, a$", fontsize=12); ax.set_title(title, fontsize=9)
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
            ax.axhline(0,           color="gray",   lw=0.5, ls="--", alpha=0.5)
            ax.axhline(V_eV,        color="black",  lw=0.8, ls=":",  alpha=0.6, label=rf"$V_g={V_eV:.3f}$ eV")
            half_U = abs(U_eV) / 2
            if half_U > 1e-4:
                ax.axhline(V_eV + half_U, color="tomato", lw=0.8, ls=":", alpha=0.8, label=rf"$\pm|U|/2={half_U*1e3:.1f}$ meV")
                ax.axhline(V_eV - half_U, color="tomato", lw=0.8, ls=":", alpha=0.8)
            ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.25)
        ax1.set_ylabel(r"$E$ (eV)", fontsize=12)

        out_dir = Path(f"plots_d=60_AB/wykresF/{tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = str(out_dir / f"wykresF_{tag}")
        plt.tight_layout()
        plt.savefig(fname + ".pdf"); plt.savefig(fname + ".png", dpi=150)
        plt.close(fig)
        print(f"  Zapisano: {fname}.png")


# ===========================================================================
# WYKRES D — mapa G(Vb, B)
# ===========================================================================
def zadanie_D_mapa_G_Vb_B():
    print("\n" + "=" * 70)
    print("WYKRES D (PLOT): Mapa G(Vb, B)")
    print("=" * 70)
    ensure_plots_dir()

    cache_file = "data_d=60_AB/wykresD_G_map.npy"
    cache_meta = "data_d=60_AB/wykresD_G_map_meta.npz"
    if not Path(cache_file).exists():
        print("  Brak cache — pomijam."); return None

    G_map    = np.load(cache_file); meta = np.load(cache_meta)
    B_phys   = meta['B']; Vb_arr = meta['Vb']
    Vt_fixed = float(np.atleast_1d(meta['Vt'])[0])

    hbar_SI = 1.0545718e-34; e_SI = 1.602176634e-19; Rc_SI = 45e-9

    if 'n_tot_SI' in meta:
        n_tot_SI = meta['n_tot_SI']
    else:
        # Fallback: szybki solver (bez transportu Kwant)
        cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
        phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
        sc = DoubleBLGSolver(cap_params, phys_params).solve_1D_sweep(Vb_arr, Vt=Vt_fixed)
        n_tot_SI = np.abs(sc['n1'] + sc['n2']) * 1e15

    k_F_SI  = np.sqrt(np.pi * n_tot_SI)
    B_line  = hbar_SI * k_F_SI / (e_SI * Rc_SI)
    B_2D, Vb_2D = np.meshgrid(B_phys, Vb_arr)

    fig, ax = plt.subplots(figsize=(9, 6))
    pcm = ax.pcolormesh(Vb_2D, B_2D, G_map.T, cmap='viridis',
                        vmin=np.min(G_map), vmax=np.max(G_map), shading='auto')
    cb  = fig.colorbar(pcm, ax=ax, pad=0.02); cb.set_label(r'$G$  (2$e^2/h$)', fontsize=11)
    mask = B_line <= B_phys.max()
    if mask.any():
        ax.plot(Vb_arr[mask], B_line[mask], 'w--', lw=2.2, label=r'$R_c = 45$ nm')
        ax.legend(fontsize=10, loc='upper left',
                  framealpha=0.6, edgecolor='w', labelcolor='w', facecolor='#333333')
    ax.set_xlabel(r'$V_b$ (V)', fontsize=12); ax.set_ylabel(r'$B$ (T)', fontsize=12)
    ax.set_title(
        r'Wykres D — Mapa przewodności $G(V_b, B)$' '\n'
        rf'$V_t = {Vt_fixed:.0f}$ V,  linia: $R_c = ħk_F / eB = 45$ nm',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("plots_d=60_AB/wykresD_mapa_G_Vb_B.pdf")
    plt.savefig("plots_d=60_AB/wykresD_mapa_G_Vb_B.png", dpi=150)
    print("Zapisano: plots_d=60_AB/wykresD_mapa_G_Vb_B.pdf")
    return G_map


# ===========================================================================
# WYKRES E — profil potencjału
# ===========================================================================
def zadanie_E_profil_potencjalu(Vt=5.0, Vb=5.0, B_T=3.0, L_nm=200.0, d_nm=0.01):
    """Narysuj wykres E tylko z danych zapisanych w data_d=60_AB/."""
    print("\n" + "=" * 70)
    print(f"WYKRES E (PLOT): Vt={Vt} V  Vb={Vb} V  B={B_T} T")
    print("=" * 70)
    ensure_plots_dir()

    cache_file = f"data_d=60_AB/wykresE_profil_potencjalu_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}.npz"
    loaded = _load_npz(cache_file)
    if loaded is None:
        print("  Brak cache — pomijam."); return None

    x_nm      = loaded["x_nm"]; B_x       = loaded["B_x"]
    V_x       = loaded["V_x"];  U_x       = loaded["U_x"]
    low_layer = loaded["low_layer"]; upp_layer = loaded["upp_layer"]
    Vg1_eV    = _s(loaded["Vg1_eV"]); U1_eV = _s(loaded["U1_eV"])
    Vg2_eV    = _s(loaded["Vg2_eV"]); U2_eV = _s(loaded["U2_eV"])
    sc_n1     = _s(loaded["n1"]);      sc_n2 = _s(loaded["n2"])
    L_nm_data = _s(loaded["L_nm"]) if "L_nm" in loaded else L_nm

    print(f"  BLG1 (x≥0): Vg1={Vg1_eV:.4f} eV,  U1={U1_eV:.4f} eV,  n1={sc_n1:.3f}×10¹⁵ m⁻²")
    print(f"  BLG2 (x<0): Vg2={Vg2_eV:.4f} eV,  U2={U2_eV:.4f} eV,  n2={sc_n2:.3f}×10¹⁵ m⁻²")

    band_top    = V_x + np.abs(U_x)/2
    band_bottom = V_x - np.abs(U_x)/2

    fig, (ax_B, ax_pot) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0.08},
    )
    fig.suptitle(
        rf"Wykres E — Pole magnetyczne i profil pasm w złączu BLG" "\n"
        rf"$V_t={Vt:.1f}$ V,  $V_b={Vb:.1f}$ V,  $B={B_T:.1f}$ T,  "
        rf"$d={d_nm:.0f}$ nm,  $C_t=C_m=2.55\times10^{{15}}$ m$^{{-2}}$V$^{{-1}}$",
        fontsize=10,
    )

    ax_B.step(x_nm, B_x, where='mid', color='darkorange', lw=2.0, label=rf'$B(x)$')
    ax_B.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_B.axvline(0, color='gray', lw=1.0, ls=':', alpha=0.7)
    ax_B.fill_between(x_nm, 0, B_x, alpha=0.15, color='darkorange')
    ax_B.set_ylabel(r'$B$ (T)', fontsize=11)
    ax_B.set_ylim(-B_T * 1.5, B_T * 1.5); ax_B.set_yticks([-B_T, 0, B_T])
    ax_B.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1f}'))
    ax_B.legend(fontsize=8, loc='upper right'); ax_B.grid(True, alpha=0.2)

    ax_pot.plot(x_nm, band_top,    color='navy',     lw=2.0, label=r'Pasmo przewodnictwa: $V+|U|/2$')
    ax_pot.plot(x_nm, band_bottom, color='firebrick', lw=2.0, label=r'Pasmo walencyjne: $V-|U|/2$')
    ax_pot.plot(x_nm, V_x,         color='black',     lw=1.5, ls='--', label=r'$V(x)$ (śr. potencjał)')
    ax_pot.fill_between(x_nm, band_bottom, band_top, alpha=0.10, color='purple',
                        label=rf'Przerwa $|U(x)|$  [tanh, $d={d_nm:.0f}$ nm]')
    ax_pot.axvline(0, color='gray', lw=1.0, ls=':', alpha=0.7, label='Granica złącza')
    ax_pot.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

    for xpos, gap, mid, col in [
        ( L_nm_data * 0.60, abs(U1_eV), Vg1_eV, 'navy'),
        (-L_nm_data * 0.60, abs(U2_eV), Vg2_eV, 'firebrick'),
    ]:
        ax_pot.annotate(
            rf'$|U|={gap*1e3:.1f}$ meV', xy=(xpos, mid), fontsize=8, color=col,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
        )

    ax_pot.set_xlabel(r'$x$ (nm)', fontsize=11)
    ax_pot.set_ylabel(r'Energia (eV)', fontsize=11)
    ax_pot.legend(fontsize=8, loc='best'); ax_pot.grid(True, alpha=0.25)
    ax_pot.set_xlim(-L_nm_data, L_nm_data)

    plt.tight_layout()
    fname = f"plots_d=60_AB/wykresE_profil_potencjalu_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}"
    plt.savefig(fname + ".pdf"); plt.savefig(fname + ".png", dpi=150)
    print(f"Zapisano: {fname}.pdf")
    return {'x_nm': x_nm, 'B_x': B_x, 'V_x': V_x, 'U_x': U_x}


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "=" * 70)
    print("SYMULACJE TRANSPORTU PRZEZ DWUWARSTWOWY GRAFEN — ETAP WYKRESÓW")
    print("=" * 70)
    ensure_plots_dir(); ensure_data_dir()

    for fn, label in [
        # (zadanie1_dyspersja,              "zadanie 1"),
        # (zadanie2_dyspersja_z_przerwa,    "zadanie 2"),
        # (zadanie3_przewodnosc_vs_B,       "zadanie 3"),
        # (zadanie4_parametry_samouzgodnione, "zadanie 4"),
        # (zadanie5_przewodnosc_samouzgodniona, "zadanie 5"),
        # (zadanie6_mapy_2D,                "zadanie 6"),
        # (zadanie8_mapa_G_Vb_B_Vt0,       "zadanie 8"),
        # (zadanie8_mapa_G_Vb_B_Vt0_z_poziomami_Landaua, "mapa G z poziomami Landaua"),
        # (lambda: wykres_distance_map_z_warunkami(Vt_target=0.0, width_nm=300.0, buffer_frac=0.0001), "mapa G(Vb, B) z warunkami"),
        # (lambda: wykres_G_od_B_z_mapy(Vb_targets=[20.0, 30.0, 45.0]), "wykres G(B) z mapy"),
        (zadanie9_prad_lokalny,            "zadanie 9"),
        # (lambda: wykres_G_od_Vb_przy_stalym_B(B_target=9.0), "wykres G(Vb) przy stałym B"),
        # # (zadanie_F_dyspersja,             "wykres F"),
        # (zadanie_A_elektrostatyka_1D,     "wykres A"),
        # (zadanie_B_mapa_2D_ntot,          "wykres B"),
        # (zadanie_C_pasma_z_B,             "wykres C"),
        # (zadanie_D_mapa_G_Vb_B,           "wykres D"),
        # (lambda: zadanie_E_profil_potencjalu(Vt=0.0, Vb=20.0, B_T=6), "wykres E"),
        # (lambda: zadanie_E_profil_potencjalu(Vt=0.0, Vb=30.0, B_T=6), "wykres E"),
        # (lambda: zadanie_E_profil_potencjalu(Vt=0.0, Vb=45.0, B_T=6), "wykres E"),
        # (lambda: wykres_G_od_B_stale_Vb(20.0, Vt_target=0.0, width_nm=300.0, buffer_frac=0.002), "wykres G(B) przy stałym Vb"),
        # (lambda: wykres_G_od_B_stale_Vb(30.0, Vt_target=0.0, width_nm=300.0, buffer_frac=0.002), "wykres G(B) przy stałym Vb"),
        # (lambda: wykres_G_od_B_stale_Vb(45.0, Vt_target=0.0, width_nm=300.0, buffer_frac=0.002), "wykres G(B) przy stałym Vb"),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"Błąd w {label}: {e}")

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO ETAP WYKRESÓW")
    print("=" * 70)


if __name__ == "__main__":
    main()
