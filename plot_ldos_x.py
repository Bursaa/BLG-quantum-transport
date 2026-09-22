"""Wykres przekroju LDOS(x) dla złącza BLG.

LDOS jest sumowany po wszystkich węzłach o tej samej współrzędnej x
(czyli po kierunku poprzecznym y oraz po obu warstwach). Parametry
samouzgodnione są zgodne z obliczeniami zadania 8.
"""

from pathlib import Path

import kwant
import matplotlib.pyplot as plt
import numpy as np

from Constants import E_scale, l_scale, T_scale
from BLG_parameters import (
    BLGPhysicsParameters,
    CapacitanceParameters,
    DoubleBLGSolver,
)
from BLG_system import (
    GAMMA1,
    T_INTRALAYER,
    BLGSystemParameters,
    make_blg_system,
)


Vt = 0.0
LDOS_CASES = ((9.0,28.3), (9.0, 30.2))
ENERGY_E_V = 0.0
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_d=0_AB"
PLOTS_DIR = PROJECT_DIR / "plots_d=0_AB"


def ldos_x_for_vb(B_T: float, Vb: float, *, energy_eV: float | None = None):
    """Oblicz LDOS zsumowany wzdłuż y dla jednego napięcia Vb."""
    cache_path = DATA_DIR / f"ldos_x_B{B_T:g}_Vb{Vb:g}.npz"
    requested_energy_eV = 0.0 if energy_eV is None else float(energy_eV)
    if cache_path.exists():
        with np.load(cache_path) as cached:
            x_cached = np.asarray(cached["x_nm"], dtype=float)
            ldos_cached = np.asarray(cached["ldos_x"], dtype=float)
            energy_cached = float(np.asarray(cached["energy_eV"]).reshape(-1)[0])
        if np.isclose(energy_cached, requested_energy_eV):
            print(f"Wczytano zapisane dane LDOS: {cache_path}")
            return x_cached, ldos_cached, energy_cached, None, -1
        print(f"Cache ma inną energię ({energy_cached} eV), obliczam ponownie.")

    cache_path = DATA_DIR / f"ldos_x_B{B_T:g}_Vb{Vb:g}.npz"
    requested_energy_eV = 0.0 if energy_eV is None else float(energy_eV)

    if cache_path.exists():
        try:
            with np.load(cache_path) as cached:
                required = {"x_nm", "ldos_x", "energy_eV", "B_T", "Vt", "Vb"}
                if not required.issubset(cached.files):
                    raise ValueError("brak wymaganych danych")
                values_match = (
                    np.isclose(float(np.asarray(cached["B_T"]).reshape(-1)[0]), B_T)
                    and np.isclose(float(np.asarray(cached["Vt"]).reshape(-1)[0]), Vt)
                    and np.isclose(float(np.asarray(cached["Vb"]).reshape(-1)[0]), Vb)
                    and np.isclose(
                        float(np.asarray(cached["energy_eV"]).reshape(-1)[0]),
                        requested_energy_eV,
                    )
                )
                if not values_match:
                    raise ValueError("niezgodne parametry cache")
                x_cached = np.asarray(cached["x_nm"], dtype=float)
                ldos_cached = np.asarray(cached["ldos_x"], dtype=float)
                if x_cached.ndim != 1 or ldos_cached.shape != x_cached.shape:
                    raise ValueError("niezgodny kształt danych")
                n_sites = (
                    int(np.asarray(cached["n_sites"]).reshape(-1)[0])
                    if "n_sites" in cached.files else -1
                )
            print(f"Wczytano zapisane dane LDOS: {cache_path}")
            return x_cached, ldos_cached, requested_energy_eV, None, n_sites
        except (OSError, ValueError, KeyError) as exc:
            print(f"Nie można użyć cache ({exc}) — obliczam ponownie.")

    cap_params = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    sc = DoubleBLGSolver(cap_params, phys_params).solve(Vt, Vb)

    # Konwencja z conductance_map_G_B_Vb: BLG1 jest obrócony,
    # dlatego U1 ma zmieniony znak w układzie Kwanta.
    V1 = sc.Vg1 / E_scale
    U1 = -sc.U1 / E_scale
    V2 = sc.Vg2 / E_scale
    U2 = sc.U2 / E_scale

    params = BLGSystemParameters(
        L=200.0 / l_scale,
        W=100.0 / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        B=B_T / T_scale,
        V1=V1,
        U1=U1,
        V2=V2,
        U2=U2,
        d=0.001 / l_scale,
        name=f"LDOS_x_Vb{Vb:g}_B{B_T:g}",
    )
    print("Buduje Układ BLG")
    blg = make_blg_system(params, use_potential_profile=True)
    print("Układ BLG zbudowany")
    energy = requested_energy_eV
    print(f"Obliczam LDOS dla energii E={energy} eV")
    density = kwant.ldos(blg.system, energy)
    print("LDOS obliczone")
    x_nm = np.array([site.pos[0] * l_scale for site in blg.system.sites])

    # Współrzędne x są wspólne dla kilku podwęzłów; zaokrąglenie
    # zabezpiecza przed rozdzieleniem identycznych pozycji przez float.
    x_key = np.round(x_nm, decimals=10)
    x_values, inverse = np.unique(x_key, return_inverse=True)
    ldos_x = np.zeros_like(x_values, dtype=float)
    np.add.at(ldos_x, inverse, density)

    return x_values, ldos_x, energy * E_scale, sc, blg.system.graph.num_nodes


def ldos_xy_for_vb(B_T: float, Vb: float, *, energy_eV: float = ENERGY_E_V):
    """Zwraca LDOS na wszystkich węzłach układu jako funkcję (x, y)."""
    cache_path = DATA_DIR / f"ldos_xy_B{B_T:g}_Vb{Vb:g}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            x_nm = np.asarray(cached["x_nm"], dtype=float)
            y_nm = np.asarray(cached["y_nm"], dtype=float)
            density = np.asarray(cached["ldos_xy"], dtype=float)
            energy_cached = float(np.asarray(cached["energy_eV"]).reshape(-1)[0])
            profile_d_cached = (
                float(np.asarray(cached["profile_d_nm"]).reshape(-1)[0])
                if "profile_d_nm" in cached.files else None
            )
        if np.isclose(energy_cached, energy_eV) and np.isclose(profile_d_cached or -1.0, 0.001):
            print(f"Wczytano zapisane dane LDOS(x,y): {cache_path}")
            return x_nm, y_nm, density, energy_cached
        print(f"Cache 2D ma niezgodne parametry profilu, obliczam ponownie: {cache_path}")

    cap_params = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    sc = DoubleBLGSolver(cap_params, phys_params).solve(Vt, Vb)
    params = BLGSystemParameters(
        L=200.0 / l_scale,
        W=100.0 / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        B=B_T / T_scale,
        V1=sc.Vg1 / E_scale,
        U1=-sc.U1 / E_scale,
        V2=sc.Vg2 / E_scale,
        U2=sc.U2 / E_scale,
        d=0.001 / l_scale,
        name=f"LDOS_xy_Vb{Vb:g}_B{B_T:g}",
    )
    print(f"Obliczam LDOS(x,y) dla Vb={Vb:g} V")
    blg = make_blg_system(params, use_potential_profile=True)
    density = kwant.ldos(blg.system, energy_eV)
    x_nm = np.array([site.pos[0] * l_scale for site in blg.system.sites])
    y_nm = np.array([site.pos[1] * l_scale for site in blg.system.sites])
    np.savez(
        cache_path,
        B_T=B_T, Vt=Vt, Vb=Vb, energy_eV=energy_eV,
        profile_d_nm=0.001,
        x_nm=x_nm, y_nm=y_nm, ldos_xy=density,
    )
    print(f"Zapisano dane LDOS(x,y): {cache_path}")
    return x_nm, y_nm, density, energy_eV


def ldos_x_from_xy(x_nm: np.ndarray, density: np.ndarray):
    """Sumuje pełne LDOS(x,y) po y, bez ponownego obliczania Kwanta."""
    x_key = np.round(np.asarray(x_nm, dtype=float), decimals=10)
    density = np.asarray(density, dtype=float)
    x_values, inverse = np.unique(x_key, return_inverse=True)
    ldos_x = np.zeros_like(x_values, dtype=float)
    np.add.at(ldos_x, inverse, density)
    return x_values, ldos_x


def main():
    out_dir = PLOTS_DIR
    data_dir = DATA_DIR
    out_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    print(f"Uruchomiono plik: {Path(__file__).resolve()}")

    # Najpierw obliczamy lub wczytujemy pełne mapy 2D dokładnie raz.
    xy_by_case = {}
    ldos_by_case = {}
    for B_T, Vb in LDOS_CASES:
        print(f"Obliczam/wczytuję LDOS(x,y): B={B_T:g} T, Vb={Vb:g} V", flush=True)
        x_xy, y_xy, ldos_xy, energy_xy = ldos_xy_for_vb(B_T, Vb)
        x_nm, ldos_x = ldos_x_from_xy(x_xy, ldos_xy)
        xy_by_case[(B_T, Vb)] = (x_xy, y_xy, ldos_xy, energy_xy)
        ldos_by_case[(B_T, Vb)] = (x_nm, ldos_x)
        print(
            f"  węzłów={len(ldos_xy)}, E_F={energy_xy:.6f} eV, "
            f"max LDOS(x)={np.max(ldos_x):.6g}"
        )

    # Z map 2D tworzymy osobne wykresy 1D LDOS(x).
    for B_T, Vb in LDOS_CASES:
        x_nm, ldos_x = ldos_by_case[(B_T, Vb)]
        energy_eV = xy_by_case[(B_T, Vb)][3]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(x_nm, ldos_x, linewidth=1.5, label=rf"$V_b={Vb:g}$ V")
        ax.set_xlabel(r"$x$ (nm)")
        ax.set_ylabel(r"LDOS$(x)$ (sum po $y$)")
        ax.set_title(
            rf"Przekrój LDOS$(x)$ dla $B={B_T:g}$ T, "
            rf"$V_t=0$ V, $V_b={Vb:g}$ V"
        )
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()

        plot_name = f"ldos_x_B{B_T:g}_Vb{Vb:g}"
        plot_base = out_dir / plot_name
        data_path = data_dir / f"ldos_x_B{B_T:g}_Vb{Vb:g}.npz"
        np.savez(
            data_path,
            B_T=B_T,
            Vt=Vt,
            Vb=Vb,
            x_nm=x_nm,
            ldos_x=ldos_x,
            energy_eV=energy_eV,
        )
        # Nie używamy Path.with_suffix(): kropka w "B3.5" zostałaby
        # potraktowana jako rozszerzenie i nazwa skróciłaby się do "B3".
        png_path = plot_base.parent / f"{plot_base.name}.png"
        pdf_path = plot_base.parent / f"{plot_base.name}.pdf"
        png_path = png_path.resolve()
        pdf_path = pdf_path.resolve()
        fig.savefig(str(png_path), dpi=180)
        fig.savefig(str(pdf_path))
        plt.close(fig)
        if not png_path.exists() or not pdf_path.exists():
            raise RuntimeError(f"Nie utworzono plików wykresu: {png_path}, {pdf_path}")
        print(f"Zapisano: {png_path}")
        print(f"Zapisano: {pdf_path}")
        print(f"Zapisano dane: {data_path}")

    # Z tych samych map zapisujemy osobne wykresy 2D LDOS(x,y).
    for B_T, Vb in LDOS_CASES:
        x_xy, y_xy, ldos_xy, energy_xy = xy_by_case[(B_T, Vb)]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        scatter = ax.scatter(
            x_xy, y_xy, c=ldos_xy, s=1.0, cmap="magma",
            linewidths=0, rasterized=True,
        )
        fig.colorbar(scatter, ax=ax, pad=0.02, label="LDOS")
        ax.set_xlabel(r"$x$ (nm)")
        ax.set_ylabel(r"$y$ (nm)")
        ax.set_title(
            rf"LDOS$(x,y)$ dla $B={B_T:g}$ T, "
            rf"$V_t=0$ V, $V_b={Vb:g}$ V, $E={energy_xy:.3f}$ eV"
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        map_base = out_dir / f"ldos_xy_B{B_T:g}_Vb{Vb:g}"
        map_png = map_base.parent / f"{map_base.name}.png"
        map_pdf = map_base.parent / f"{map_base.name}.pdf"
        fig.savefig(str(map_png.resolve()), dpi=180)
        fig.savefig(str(map_pdf.resolve()))
        plt.close(fig)
        print(f"Zapisano mapę 2D: {map_png.resolve()}")
        print(f"Zapisano mapę 2D: {map_pdf.resolve()}")

    # Wykres różnicowy pomiędzy dwoma nowymi przypadkami.
    (B_low, Vb_low), (B_high, Vb_high) = LDOS_CASES
    x_low, ldos_low = ldos_by_case[(B_low, Vb_low)]
    x_high, ldos_high = ldos_by_case[(B_high, Vb_high)]
    if not np.array_equal(x_low, x_high):
        raise ValueError("Siatki x dla obu przypadków LDOS są różne.")
    delta_ldos = ldos_high - ldos_low

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_low, delta_ldos, color="darkorange", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"$x$ (nm)")
    ax.set_ylabel(r"$\Delta$LDOS$(x)$ (sum po $y$)")
    ax.set_title(rf"Różnica LDOS$(x)$: ({B_high:g} T, {Vb_high:g} V) - "
                 rf"({B_low:g} T, {Vb_low:g} V)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    difference_base = out_dir / "ldos_x_difference_B8.55_Vb25.65_minus_B8.4_Vb25.2"
    difference_data = data_dir / "ldos_x_difference_B8.55_Vb25.65_minus_B8.4_Vb25.2.npz"
    np.savez(
        difference_data,
        Vt=Vt,
        B_low=B_low,
        B_high=B_high,
        Vb_low=Vb_low,
        Vb_high=Vb_high,
        x_nm=x_low,
        delta_ldos=delta_ldos,
    )
    difference_png = difference_base.parent / f"{difference_base.name}.png"
    difference_pdf = difference_base.parent / f"{difference_base.name}.pdf"
    fig.savefig(str(difference_png.resolve()), dpi=180)
    fig.savefig(str(difference_pdf.resolve()))
    plt.close(fig)
    print(f"Zapisano wykres różnicowy: {difference_png.resolve()}")
    print(f"Zapisano wykres różnicowy: {difference_pdf.resolve()}")
    print(f"Zapisano dane różnicowe: {difference_data.resolve()}")


if __name__ == "__main__":
    main()