"""
Główny skrypt do symulacji transportu przez dwuwarstwowy grafen (bilayer graphene).

Ten skrypt demonstruje:
1. Obliczanie relacji dyspersji dla BLG
2. Przewodność w funkcji pola magnetycznego
3. Przewodność w funkcji napięcia bramkowego z parametrami samouzgodnionymi
4. Wizualizację rozkładu prądu

Autorzy: [Twoje dane]
Data: 2026
"""
import kwant
import numpy as np
import matplotlib.pyplot as plt

# Import modułów projektu
from Constants import l_scale, E_scale, T_scale, k_arr, B_arr
from BLG_system import (
    BLGSystemParameters, BilayerGrapheneSystem,
    make_blg_system, T_INTRALAYER, GAMMA1, A_GRAPHENE
)
from BLG_parameters import (
    DoubleBLGSolver, CapacitanceParameters, BLGPhysicsParameters
)
from BLG_transport import (
    calculate_dispersion,
    conductance_vs_magnetic_field,
    conductance_with_selfconsistent_params,
    conductance_map_G_B_Vb,
    calculate_total_bond_current,
    ensure_data_dir,
    ensure_plots_dir,
    calculate_conductance,
)

# Ustawienia wykresu
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (8, 6)

def plot_map(Vb_list, B_list, G_map):

    plt.figure(figsize=(8, 6))

    plt.imshow(
        G_map,
        origin="lower",
        aspect="auto",
        extent=[Vb_list.min(), Vb_list.max(),
                B_list.min(), B_list.max()],
        cmap="viridis"
    )

    plt.colorbar(label=r"G (2e²/h)")

    plt.xlabel(r"$V_b$ [V]")
    plt.ylabel(r"$B$ [T]")
    plt.title("Mapa konduktancji BLG")

    plt.tight_layout()
    plt.show()

def zadanie1_dyspersja():
    """
    Zadanie 1: Oblicz i porównaj relacje dyspersji dla BLG.
    
    Porównanie:
    - s_f = 1 (oryginalna siatka)
    - s_f = 4 (siatka przeskalowana)
    """
    print("\n" + "=" * 70)
    print("ZADANIE 1: Relacja dyspersji dwuwarstwowego grafenu")
    print("=" * 70)
    
    ensure_plots_dir()
    
    # Wspólne parametry
    common = {
        'L': 30.0 / l_scale,
        'W': 15.0 / l_scale,
        't': T_INTRALAYER,
        'gamma1': GAMMA1,
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # System 1: s_f = 1
    params1 = BLGSystemParameters(
        **common, s_f=1.0, U1=0.0, name="BLG_sf1"
    )
    blg1 = make_blg_system(params1)
    E1 = calculate_dispersion(blg1)
    
    axes[0].plot(k_arr, E1 * E_scale, 'b-', linewidth=0.5)
    axes[0].set_xlabel(r"$k$ (1/nm)")
    axes[0].set_ylabel(r"$E$ (eV)")
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].set_title(r"BLG $s_f = 1$ (oryginalna siatka)")
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    
    # System 2: s_f = 4
    params2 = BLGSystemParameters(
        **common, s_f=4.0, U1=0.0, name="BLG_sf4"
    )
    blg2 = make_blg_system(params2)
    E2 = calculate_dispersion(blg2)
    
    axes[1].plot(k_arr, E2 * E_scale, 'r-', linewidth=0.5)
    axes[1].set_xlabel(r"$k$ (1/nm)")
    axes[1].set_ylabel(r"$E$ (eV)")
    axes[1].set_ylim(-0.3, 0.3)
    axes[1].set_title(r"BLG $s_f = 4$ (przeskalowana siatka)")
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/zadanie1_dyspersja_BLG.pdf")
    plt.savefig("plots/zadanie1_dyspersja_BLG.png", dpi=150)
    print("Zapisano: plots/zadanie1_dyspersja_BLG.pdf")
    # plt.show()


def zadanie2_dyspersja_z_przerwa():
    """
    Zadanie 2: Relacja dyspersji z niezerową przerwą energetyczną U.
    """
    print("\n" + "=" * 70)
    print("ZADANIE 2: Relacja dyspersji BLG z przerwą energetyczną")
    print("=" * 70)
    
    ensure_plots_dir()
    
    params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0 / l_scale,
        V1=0.0 / E_scale,  # potencjał
        B=1.5 / T_scale,    # pole magnetyczne
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        U1=0.1 / E_scale,   # przerwa energetyczna
        d=5.0 / l_scale,
        name="BLG_z_przerwa"
    )
    
    blg = make_blg_system(params)
    E = calculate_dispersion(blg, k_points = k_arr/16)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k_arr, E * E_scale, 'b-', linewidth=0.5)
    ax.set_xlabel(r"$k$ (1/nm)")
    ax.set_ylabel(r"$E$ (eV)")
    ax.set_ylim(-0.3, 0.3)
    ax.set_title(f"BLG z U = {params.U1 * E_scale:.2f} eV, B = {params.B * T_scale:.1f} T")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(params.U1 * E_scale / 2, color='red', linestyle=':', alpha=0.7, label=f'+U/2')
    ax.axhline(-params.U1 * E_scale / 2, color='red', linestyle=':', alpha=0.7, label=f'-U/2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/zadanie2_dyspersja_z_przerwa.pdf")
    plt.savefig("plots/zadanie2_dyspersja_z_przerwa.png", dpi=150)
    print("Zapisano: plots/zadanie2_dyspersja_z_przerwa.pdf")
    # plt.show()


def zadanie3_przewodnosc_vs_B():
    """
    Zadanie 3: Przewodność w funkcji pola magnetycznego.
    """
    print("\n" + "=" * 70)
    print("ZADANIE 3: Przewodność BLG vs pole magnetyczne")
    print("=" * 70)
    
    ensure_plots_dir()
    
    params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0 / l_scale,
        V1=0.1 / E_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        U1=0.1 / E_scale,
        d=5.0 / l_scale,
        name="BLG_G_vs_B"
    )
    
    B_values = np.arange(1.0, 3.05, 0.1) / T_scale
    print(f"Obliczanie przewodności dla {len(B_values)} wartości B...")
    
    G = conductance_vs_magnetic_field(params, B_values)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(B_values * T_scale, G, 'bo-', markersize=4)
    ax.set_xlabel(r"$B$ (T)")
    ax.set_ylabel(r"$G$ ($2e^2/h$)")
    ax.set_title("Przewodność dwuwarstwowego grafenu vs pole magnetyczne")
    ax.grid(True, alpha=0.3)
    
    # Zaznacz ekstrema
    from scipy.signal import argrelextrema
    maxima = argrelextrema(G, np.greater)[0]
    minima = argrelextrema(G, np.less)[0]
    
    if len(maxima) > 0:
        ax.scatter(B_values[maxima] * T_scale, G[maxima], 
                  color='green', s=100, zorder=5, label='Maksima')
    if len(minima) > 0:
        ax.scatter(B_values[minima] * T_scale, G[minima],
                  color='red', s=100, zorder=5, label='Minima')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("plots/zadanie3_G_vs_B.pdf")
    plt.savefig("plots/zadanie3_G_vs_B.png", dpi=150)
    print("Zapisano: plots/zadanie3_G_vs_B.pdf")
    # plt.show()
    
    return G, B_values


def zadanie4_parametry_samouzgodnione():
    """
    Zadanie 4: Rozwiązanie równań samouzgodnionych dla podwójnego bilayera.
    """
    print("\n" + "=" * 70)
    print("ZADANIE 4: Parametry samouzgodnione podwójnego dwuwarstwowego grafenu")
    print("=" * 70)
    
    ensure_plots_dir()
    
    # Parametry pojemnościowe z eksperymentu
    cap_params = CapacitanceParameters(
        Cm=2.55,    # między bilayerami (hBN 80nm)
        Cb=0.653,   # bramka dolna (SiO2 330nm)
        Ct=0.0,     # brak bramki górnej
        Cg=460.0,   # między warstwami w bilayerze
    )
    
    phys_params = BLGPhysicsParameters(
        gamma1=0.39,
        hvf2=6.39**2,
        n_t=118.57,
    )
    
    solver = DoubleBLGSolver(cap_params, phys_params)
    
    # Sweep po napięciu bramki dolnej
    Vb_arr = np.linspace(-60, 60, 121)
    results = solver.solve_1D_sweep(Vb_arr, Vt=0.0)
    
    # Wykresy parametrów samouzgodnionych
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gęstość nośników
    axes[0, 0].plot(Vb_arr, results['n1'], 'b-', label=r'$n_1$ (górny BLG)')
    axes[0, 0].plot(Vb_arr, results['n2'], 'r--', label=r'$n_2$ (dolny BLG)')
    axes[0, 0].set_xlabel(r'$V_b$ (V)')
    axes[0, 0].set_ylabel(r'$n$ ($10^{15}$ m$^{-2}$)')
    axes[0, 0].set_title('Gęstość nośników')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Przerwa energetyczna U
    axes[0, 1].plot(Vb_arr, results['U1'], 'b-', label=r'$U_1$')
    axes[0, 1].plot(Vb_arr, results['U2'], 'r--', label=r'$U_2$')
    axes[0, 1].set_xlabel(r'$V_b$ (V)')
    axes[0, 1].set_ylabel(r'$U$ (eV)')
    axes[0, 1].set_title('Przerwa energetyczna (asymmetry gap)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Potencjał bramkowy
    axes[1, 0].plot(Vb_arr, results['Vg1'], 'b-', label=r'$V_{G1}$')
    axes[1, 0].plot(Vb_arr, results['Vg2'], 'r--', label=r'$V_{G2}$')
    axes[1, 0].set_xlabel(r'$V_b$ (V)')
    axes[1, 0].set_ylabel(r'$V_G$ (V)')
    axes[1, 0].set_title('Potencjał bramkowy bilayerów')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Asymetria gęstości
    axes[1, 1].plot(Vb_arr, results['dn1'], 'b-', label=r'$\Delta n_1$')
    axes[1, 1].plot(Vb_arr, results['dn2'], 'r--', label=r'$\Delta n_2$')
    axes[1, 1].set_xlabel(r'$V_b$ (V)')
    axes[1, 1].set_ylabel(r'$\Delta n$ ($10^{15}$ m$^{-2}$)')
    axes[1, 1].set_title('Asymetria gęstości między warstwami')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("plots/zadanie4_parametry_samouzgodnione.pdf")
    plt.savefig("plots/zadanie4_parametry_samouzgodnione.png", dpi=150)
    print("Zapisano: plots/zadanie4_parametry_samouzgodnione.pdf")
    # plt.show()
    
    return results


def zadanie5_przewodnosc_samouzgodniona():
    """
    Zadanie 5: Przewodność z samouzgodnionymi parametrami.
    """
    print("\n" + "=" * 70)
    print("ZADANIE 5: Przewodność BLG z parametrami samouzgodnionymi")
    print("=" * 70)
    
    ensure_plots_dir()
    
    params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0 / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        name="BLG_selfconsistent"
    )
    
    # Mniejszy zakres dla szybszych obliczeń
    Vb_arr = np.linspace(-30, 30, 31)
    
    print(f"Obliczanie przewodności dla {len(Vb_arr)} wartości Vb...")
    print("(To może zająć kilka minut...)")
    
    G, sc_results = conductance_with_selfconsistent_params(
        params, Vb_arr, Vt=0.0
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Przewodność
    axes[0, 0].plot(Vb_arr, G, 'ko-', markersize=4)
    axes[0, 0].set_xlabel(r'$V_b$ (V)')
    axes[0, 0].set_ylabel(r'$G$ ($2e^2/h$)')
    axes[0, 0].set_title('Przewodność przez złącze (junction)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Przerwa U2 (dolny bilayer)
    axes[0, 1].plot(Vb_arr, sc_results['U2'], 'r-o', markersize=3)
    axes[0, 1].set_xlabel(r'$V_b$ (V)')
    axes[0, 1].set_ylabel(r'$U_2$ (eV)')
    axes[0, 1].set_title('Przerwa energetyczna dolnego bilayera')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gęstość n2
    axes[1, 0].plot(Vb_arr, sc_results['n2'], 'b-o', markersize=3)
    axes[1, 0].set_xlabel(r'$V_b$ (V)')
    axes[1, 0].set_ylabel(r'$n_2$ ($10^{15}$ m$^{-2}$)')
    axes[1, 0].set_title('Gęstość nośników w dolnym bilayerze')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # G vs U2
    axes[1, 1].scatter(sc_results['U2'], G, c=Vb_arr, cmap='coolwarm')
    axes[1, 1].set_xlabel(r'$U_2$ (eV)')
    axes[1, 1].set_ylabel(r'$G$ ($2e^2/h$)')
    axes[1, 1].set_title(r'Przewodność vs przerwa (kolor = $V_b$)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/zadanie5_przewodnosc_samouzgodniona.pdf")
    plt.savefig("plots/zadanie5_przewodnosc_samouzgodniona.png", dpi=150)
    print("Zapisano: plots/zadanie5_przewodnosc_samouzgodniona.pdf")
    # plt.show()
    
    return G, sc_results


def zadanie6_mapy_2D():
    """
    Zadanie 6: Mapy 2D parametrów samouzgodnionych w funkcji Vtg i Vbg.

    Wykresy jako kolorowe mapy (colormaps):
      - U2  (Uq)  – przerwa energetyczna dolnego bilayera
      - n = n1+n2  – całkowita gęstość nośników
      - nt = n1    – gęstość górnego bilayera
      - nb = n2    – gęstość dolnego bilayera
      - Band offset = Vg1 - Vg2  – przesunięcie pasm między bilayerami
      - U1         – przerwa energetyczna górnego bilayera

    Zakres:
      Vtg: -6 … +6 V  (bramka górna)
      Vbg: -50 … +50 V (bramka dolna)

    Uwaga: wymaga Ct > 0, żeby Vtg miało wpływ na układ.
    Ct = 2.55e15 m^-2 V^-1 odpowiada hBN ~80 nm nad górnym bilayerem (ε=3.7).
    """
    print("\n" + "=" * 70)
    print("ZADANIE 6: Mapy 2D parametrów samouzgodnionych (Vtg, Vbg)")
    print("=" * 70)

    ensure_plots_dir()
    ensure_data_dir()

    # --- Parametry pojemnościowe ---
    cm = CapacitanceParameters.calculate_capacitance(epsilon_r=3.3, d_nm=1)
    print(cm)
    cap_params = CapacitanceParameters(
        Cm=cm,   # między bilayerami (hBN 80 nm)
        Cb=0.653,  # bramka dolna (SiO2 330 nm)
        Ct=2.55,   # bramka górna (hBN ~80 nm) — zmień wg geometrii próbki
        Cg=460.0,  # między warstwami w bilayerze
    )

    phys_params = BLGPhysicsParameters(
        gamma1=0.39,
        hvf2=6.39**2,
        n_t=118.57,
    )

    solver = DoubleBLGSolver(cap_params, phys_params)

    # --- Siatka napięć ---
    Vtg_arr = np.arange(-6.0, 6.1, 0.5)   # 25 punktów
    Vbg_arr = np.arange(-6.0, 6.1, 0.5)  # 51 punktów
    n_Vtg, n_Vbg = len(Vtg_arr), len(Vbg_arr)
    print(f"Siatka: {n_Vtg} × {n_Vbg} = {n_Vtg * n_Vbg} punktów")

    # --- Plik cache — nie obliczaj ponownie jeśli dane istnieją ---
    cache_file = "data/zadanie6_mapy_2D.npy"
    cache_meta = f"data/zadanie6_mapy_2D_meta.npz"
    try:
        res = np.load(cache_file, allow_pickle=True).item()
        meta = np.load(cache_meta)
        # Sprawdź czy siatka się zgadza
        if (np.allclose(meta['Vtg'], Vtg_arr) and
                np.allclose(meta['Vbg'], Vbg_arr)):
            print("Wczytano wyniki z cache.")
        else:
            raise ValueError("Inna siatka — przeliczam.")
    except Exception:
        print("Obliczam parametry samouzgodnione...")
        res = solver.solve_sweep(Vtg_arr, Vbg_arr)
        np.save(cache_file, res)
        np.savez(cache_meta, Vtg=Vtg_arr, Vbg=Vbg_arr)
        print("Zapisano wyniki do cache.")

    # --- Siatki do pcolormesh (osie: x=Vtg, y=Vbg) ---
    # res[key] ma kształt (n_Vtg, n_Vbg), więc transponujemy do (n_Vbg, n_Vtg)
    Vtg_2D, Vbg_2D = np.meshgrid(Vtg_arr, Vbg_arr)  # shape (n_Vbg, n_Vtg)

    def _map(key):
        return res[key].T  # transpose → (n_Vbg, n_Vtg)

    U2_map    = _map('U2')
    U1_map    = _map('U1')
    n1_map    = _map('n1')
    n2_map    = _map('n2')
    ntot_map  = n1_map + n2_map
    boff_map  = _map('Vg1') - _map('Vg2')  # band offset

    # --- Rysowanie ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(r"Parametry samouzgodnione podwójnego bilayera: "
                 r"$C_t = C_m = 2.55\times10^{15}$ m$^{-2}$V$^{-1}$",
                 fontsize=11)

    # Ostatnia wartość True = narysuj kontur U=0 (linia zamknięcia przerwy)
    datasets = [
        (axes[0, 0], U2_map,   r"$U_2$ (eV)",               "RdBu_r",
         r"$U_q = U_2$ — przerwa dolny BLG",   True),
        (axes[0, 1], ntot_map, r"$n_t + n_b$ ($10^{15}$ m$^{-2}$)", "RdBu_r",
         r"$n = n_t + n_b$ — całkowita gęstość", False),
        (axes[0, 2], n1_map,   r"$n_t$ ($10^{15}$ m$^{-2}$)",       "RdBu_r",
         r"$n_t = n_1$ — gęstość górny BLG",    False),
        (axes[1, 0], n2_map,   r"$n_b$ ($10^{15}$ m$^{-2}$)",       "RdBu_r",
         r"$n_b = n_2$ — gęstość dolny BLG",    False),
        (axes[1, 1], boff_map, r"$V_{G1} - V_{G2}$ (V)",            "PiYG",
         r"Band offset $= V_{G1} - V_{G2}$",    False),
        (axes[1, 2], U1_map,   r"$U_1$ (eV)",               "RdBu_r",
         r"$U_1$ — przerwa górny BLG",           True),
    ]

    for ax, data, cblabel, cmap_name, title, add_gap_contour in datasets:
        vmax = np.max(np.abs(data))
        vmin = -vmax if vmax > 0 else -1e-9
        pcm = ax.pcolormesh(Vtg_2D, Vbg_2D, data,
                            cmap=cmap_name, vmin=vmin, vmax=vmax,
                            shading='auto')
        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(cblabel, fontsize=8)
        ax.set_xlabel(r"$V_{tg}$ (V)")
        ax.set_ylabel(r"$V_{bg}$ (V)")
        ax.set_title(title, fontsize=9)
        # Linia zerowa na osi x i y
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
        ax.axvline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
        # Kontur U=0 — linia zamknięcia przerwy energetycznej
        if add_gap_contour and vmax > 1e-6:
            cs = ax.contour(Vtg_2D, Vbg_2D, data, levels=[0],
                            colors='k', linewidths=1.5)
            ax.clabel(cs, fmt=r"$U=0$", fontsize=7, inline=True)

    plt.tight_layout()
    plt.savefig("plots/zadanie6_mapy_2D.pdf")
    plt.savefig("plots/zadanie6_mapy_2D.png", dpi=150)
    print("Zapisano: plots/zadanie6_mapy_2D.pdf")
    # plt.show()

    return res


def zadanie8_mapa_G_Vb_B_Vt0():
    """
    Zadanie 8: Mapa przewodności G(Vb, B) dla stałego Vt = 0 V.

    Plateau w G przy konkretnych B wskazują kwantowanie Landaua.
    Skończona przewodność wzdłuż linii n=0 przy B≠0 to sygnatura stanów wężowych.

    Osie: x = B [T], y = Vb [V], kolor = G [2e²/h]
    Siatka: 51 B × 101 Vb = 5151 pkt (z cache)
    """
    print("\n" + "=" * 70)
    print("ZADANIE 8: Mapa G(Vb, B) dla Vt = 0 V — sygnatura stanów wężowych")
    print("=" * 70)

    ensure_plots_dir()
    ensure_data_dir()

    # Zakresy i kroki (zgodnie z wcześniejszymi ustaleniami)
    Vb_arr = np.arange(0.0, 50.0, 0.5)  # 101 wartości Vb [V]
    B_phys = np.arange(0.0, 10.0, 0.2)  # 51 wartości B  [T]
    Vt_value = 0.0  # Brak zależności od Vt

    cap_params = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)

    base_params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0 / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        name="BLG_z8_Vt0",
        d=5.0 / l_scale
    )

    n_B, n_Vb = len(B_phys), len(Vb_arr)
    print(f"Siatka: {n_B} B × {n_Vb} Vb = {n_B * n_Vb} pkt")

    # --- Cache (zmienione nazwy dla Vt = 0) ---
    cache_file = "data/zadanie8_G_map_Vt0.npy"
    cache_meta = "data/zadanie8_G_map_meta_Vt0.npz"
    G_map = None

    try:
        G_map = np.load(cache_file)
        meta = np.load(cache_meta)
        if (
            G_map.shape != (n_B, n_Vb)
            or not np.allclose(meta["B"], B_phys)
            or not np.allclose(meta["Vb"], Vb_arr)
            or not np.isclose(meta["Vt"], Vt_value)
        ):
            raise ValueError("Niezgodna siatka lub wartość Vt.")
        print("Wczytano z cache.")
    except Exception as exc:
        print(f"Cache nieważny ({exc}), obliczam...")

        # Obliczenie pełnej mapy 2D dla Vt = 0
        G_map = conductance_map_G_B_Vb(
            base_params,
            B_phys,
            Vb_arr,
            Vt=Vt_value,
            cap_params=cap_params,
            phys_params=phys_params,
            verbose=True,
        )

        # Zapis do cache
        np.save(cache_file, G_map)
        np.savez(cache_meta, B=B_phys, Vb=Vb_arr, Vt=Vt_value)
        print("Obliczenia zakończone i zapisane do cache.")

    # --- Wykres: Pojedyncza mapa 2D ---
    B_2D, Vb_2D = np.meshgrid(B_phys, Vb_arr)  # shape (n_Vb, n_B)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Transpozycja G_map z (n_B, n_Vb) na (n_Vb, n_B) dopasowana do meshgrid
    pcm = ax.pcolormesh(B_2D, Vb_2D, G_map.T, cmap="viridis", shading="auto")

    # Dodanie paska koloru (colorbar)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.03)
    cbar.set_label(r"Przewodność $G$ ($2e^2/h$)", fontsize=11)

    # Opisy osi i tytuł
    ax.set_xlabel(r"Pole magnetyczne $B$ (T)", fontsize=11)
    ax.set_ylabel(r"Napięcie $V_b$ (V)", fontsize=11)
    ax.set_title(
        r"Przewodność $G(V_b, B)$ złącza BLG dla $V_t = 0$ V"
        "\n"
        r"(parametry samouzgodnione, $C_t = C_m = 2.55\times10^{15}$ m$^{-2}$V$^{-1}$)",
        fontsize=12,
        pad=15,
    )

    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Zapisywanie wykresu
    plt.savefig("plots/zadanie8_G_Vb_B_Vt0.pdf")
    plt.savefig("plots/zadanie8_G_Vb_B_Vt0.png", dpi=150)
    print("Zapisano: plots/zadanie8_G_Vb_B_Vt0.pdf oraz .png")

    # plt.show()

    return G_map


def wykres_G_od_B_stale_Vb(Vb_target, Vt_target=0.0):
    """
    Program generujący gęsty wykres 1D przewodności G(B) 
    dla zadanego Vb oraz Vt w zakresie B od 0 do 10 T (200 punktów).
    """
    print("\n" + "=" * 70)
    print(f"WYKRES 1D: G(B) dla Vb = {Vb_target} V, Vt = {Vt_target} V")
    print("=" * 70)

    ensure_plots_dir()
    ensure_data_dir()

    # Generowanie dokładnie 100 punktów od 0.0 do 10.0 T
    B_phys = np.linspace(0.0, 10.0, 50)
    Vb_arr = np.array([Vb_target])  # Jednoelementowa tablica dla funkcji mapującej

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)

    base_params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0  / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        name=f"BLG_z8_line_Vb{Vb_target}_Vt{Vt_target}",
        d=5.0 / l_scale
    )

    # Unikalna nazwa pliku cache dla wybranych wartości bramkowania
    cache_file = f"data/zadanie8_line_Vb_{Vb_target}_Vt_{Vt_target}.npz"
    G_values = None

    try:
        data = np.load(cache_file)
        # Sprawdzenie, czy gęstość punktów w cache się zgadza
        if len(data['B']) != 50:
            raise ValueError("Inna liczba punktów w cache.")
        G_values = data['G']
        B_phys = data['B']
        print("Wczytano gęstą linię G(B) z cache.")
    except Exception as exc:
        print(f"Brak cache ({exc}), obliczam 100 punktów...")
        
        # Wywołujemy istniejącą funkcję – zwróci macierz o wymiarach (n_B, n_Vb) -> (100, 1)
        G_map = conductance_map_G_B_Vb(
            base_params, B_phys, Vb_arr, Vt=Vt_target,
            cap_params=cap_params, phys_params=phys_params,
            verbose=True
        )
        # Wyciągamy wektor 1D dla naszego jednego Vb
        G_values = G_map[:, 0]
        
        # Zapis do cache
        np.savez(cache_file, G=G_values, B=B_phys, Vb=Vb_target, Vt=Vt_target)
        print("Zapisano obliczenia do cache.")

    # --- Rysowanie wykresu 1D ---
    plt.figure(figsize=(10, 6))
    plt.plot(B_phys, G_values, '-', marker='o', markersize=4, color='teal', linewidth=1.5, label=r'$G(B)$')
    
    # Formatowanie wykresu
    plt.title(
        r"Przewodność $G(B)$ dla złącza BLG" "\n"
        rf"(ustalone $V_b = {Vb_target}$ V, $V_t = {Vt_target}$ V)", 
        fontsize=12, pad=15
    )
    plt.xlabel(r"Pole magnetyczne $B$ (T)", fontsize=11)
    plt.ylabel(r"Przewodność $G$ ($2e^2/h$)", fontsize=11)
    
    plt.xlim(0.0, 10.0)
    # Automatyczne dopasowanie osi Y z lekkim marginesem
    plt.ylim(bottom=0.0, top=np.max(G_values) * 1.1) 
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Zapis do plików
    filename_base = f"plots/zadanie8_G_B_Vb_{Vb_target}_Vt_{Vt_target}"
    plt.savefig(f"{filename_base}.png", dpi=150)
    plt.savefig(f"{filename_base}.pdf")
    print(f"Wykres zapisano jako: {filename_base}.png/.pdf")
    
    # plt.show()
    return B_phys, G_values

def zadanie9_prad_lokalny(Vt: float = 5.0, Vb: float = 5.0, B_T: float = 3.0):
    """
    Zadanie 9: Rozkład prądu lokalnego w złączu BLG w polu magnetycznym.

    Stany wężowe widoczne jako prąd płynący zygzakiem wzdłuż granicy
    między regionami n i p (gdzie gęstość nośników zmienia znak).

    Args:
        Vt:  napięcie bramki górnej [V]
        Vb:  napięcie bramki dolnej [V]
        B_T: pole magnetyczne [T]
    """
    print("\n" + "=" * 70)
    print(f"ZADANIE 9: Prąd lokalny  Vt={Vt}V  Vb={Vb}V  B={B_T}T")
    print("=" * 70)

    ensure_plots_dir()
    ensure_data_dir()

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)

    # Parametry samouzgodnione dla danego punktu (Vt, Vb)
    sc = solver.solve(Vt, Vb)
    V1 =  sc.Vg1 / E_scale
    U1 = -sc.U1  / E_scale  # negacja: BLG1 obrócony — top layer → bottom layer w Kwancie
    V2 =  sc.Vg2 / E_scale
    U2 =  sc.U2  / E_scale

    print(f"  Parametry samouzgodnione:")
    print(f"    BLG1 (prawa):  Vg1={sc.Vg1:.3f} V,  U1_kwant={U1*E_scale:.4f} eV (solver: {sc.U1:.4f}),  n1={sc.n1:.3f}")
    print(f"    BLG2 (lewa):   Vg2={sc.Vg2:.3f} V,  U2      ={U2*E_scale:.4f} eV,  n2={sc.n2:.3f}")
    print(f"    U w x=0 (tanh midpoint): {(U1+U2)/2*E_scale:.4f} eV  {'→ przerwa domknięta' if abs((U1+U2)/2) < 1e-4 else ''}")

    params = BLGSystemParameters(
        L=50.0 / l_scale,
        W=25.0  / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        B=B_T / T_scale,
        V1=0.1/E_scale, U1=0.01/E_scale,
        V2=0.1/E_scale, U2=0.01/E_scale,
        name=f"z9_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}",
        d=10.0 / l_scale
    )
    blg = make_blg_system(params, use_potential_profile=True)
    energy = 0.0
    print(f"  Energia transportu: {energy * E_scale:.4f} eV")

    # Sumaryczny prąd wiązań po wszystkich modach z ołowiu 0
    J_mods = calculate_total_bond_current(blg, energy)
    if len(J_mods) == 0:
        print("  Brak propagujących modów przy tej energii — pomijam.")
        return None, blg

    G = calculate_conductance(blg, energy)
    print(f"  Przewodność G = {G:.3f}  (2e²/h)")
    for i, J in enumerate(J_mods):
        print(f"  Wiązania: {len(J)},  max|J| = {np.max(np.abs(J)):.4f}")

        # --- Wykres prądu lokalnego ---
        label = rf"$V_t={Vt:.0f}$ V, $V_b={Vb:.0f}$ V, $B={B_T:.0f}$ T,  $G={G:.2f}\ (2e^2/h), mod={i}$"

        fig, ax = plt.subplots(figsize=(13, 5))
        kwant.plotter.current(
            blg.system, J,
            ax=ax,
            colorbar=True,
            show=False,
            cmap='RdBu_r',
        )
        ax.set_title(
            rf"Prąd lokalny w złączu BLG — {label}"
            "\n"
            r"(czerwony: w prawo, niebieski: w lewo; prąd wzdłuż $x=0$ → stany wężowe)"
        )
        ax.set_xlabel(r"$x$ (j.a.)")
        ax.set_ylabel(r"$y$ (j.a.)")
        ax.axvline(0, color='k', linewidth=1.0, linestyle='--', alpha=0.6,
                label=r"granica złącza $x=0$")
        ax.legend(fontsize=8, loc='upper right')

        fname = f"plots/zadanie9_current_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}_mod{i}_static_potential"
        plt.tight_layout()
        plt.savefig(fname + ".pdf")
        plt.savefig(fname + ".png", dpi=150)
        print(f"Zapisano: {fname}.pdf")
    
    # plt.show()

    return J, blg


def zadanie_A_elektrostatyka_1D():
    """
    Wykres A: Elektrostatyka 1D podwójnego bilayera.

    Oś X : Vb ∈ [-60, 60] V (bramka dolna), Vt = 0.
    Lewa oś Y  : gęstości n1 (górny BLG) i n2 (dolny BLG) [10^15 m^-2].
    Prawa oś Y : przerwy energetyczne U1, U2 [eV] (twinx).
    """
    print("\n" + "=" * 70)
    print("WYKRES A: Elektrostatyka 1D — n1, n2 oraz U1, U2 vs Vb")
    print("=" * 70)

    ensure_plots_dir()

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)

    Vb_arr = np.linspace(-60, 60, 241)
    print(f"Sweep {len(Vb_arr)} pkt Vb przy Vt = 0 V …")
    res = solver.solve_1D_sweep(Vb_arr, Vt=0.0)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    lns1 = ax1.plot(Vb_arr, res['n1'], 'b-',  lw=1.8, label=r'$n_1$ (górny BLG)')
    lns2 = ax1.plot(Vb_arr, res['n2'], 'r--', lw=1.8, label=r'$n_2$ (dolny BLG)')
    ax1.set_xlabel(r'$V_b$ (V)', fontsize=12)
    ax1.set_ylabel(r'Gęstość $n$  ($10^{15}$ m$^{-2}$)', fontsize=12)
    ax1.axhline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
    ax1.axvline(0, color='gray', lw=0.6, ls='--', alpha=0.6)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(-60, 60)

    ax2 = ax1.twinx()
    lns3 = ax2.plot(Vb_arr, res['U1'], 'b:',  lw=1.8, label=r'$U_1$ (górny BLG)')
    lns4 = ax2.plot(Vb_arr, res['U2'], 'r-.', lw=1.8, label=r'$U_2$ (dolny BLG)')
    ax2.set_ylabel(r'Przerwa energetyczna $U$ (eV)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.axhline(0, color='gray', lw=0.3, ls='--', alpha=0.4)

    lns  = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=9)

    ax1.set_title(
        r'Wykres A — Elektrostatyka 1D podwójnego BLG'
        '\n'
        r'$V_t = 0$ V,  $C_t = C_m = 2.55\times10^{15}$ m$^{-2}$V$^{-1}$',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("plots/wykresA_elektrostatyka_1D.pdf")
    plt.savefig("plots/wykresA_elektrostatyka_1D.png", dpi=150)
    print("Zapisano: plots/wykresA_elektrostatyka_1D.pdf")
    # plt.show()
    return res


def zadanie_B_mapa_2D_ntot():
    """
    Wykres B: Elektrostatyka 2D.

    Kolormapa całkowitej gęstości n_tot = n1+n2 jako funkcji Vt i Vb (oboje
    w zakresie [-60, 60] V).  Nałożone linie konturowe U1 = 0 (czarny)
    i U2 = 0 (zielony) zaznaczają granicę zamknięcia przerwy energetycznej.
    """
    print("\n" + "=" * 70)
    print("WYKRES B: Mapa 2D n_tot(Vt, Vb) z konturami U1=0, U2=0")
    print("=" * 70)

    ensure_plots_dir()
    ensure_data_dir()

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)

    Vt_arr = np.arange(-60.0, 61.0, 3.0)   # 41 punktów
    Vb_arr = np.arange(-60.0, 61.0, 3.0)   # 41 punktów

    cache_file = "data/wykresB_ntot.npy"
    cache_meta = "data/wykresB_ntot_meta.npz"
    try:
        res = np.load(cache_file, allow_pickle=True).item()
        meta = np.load(cache_meta)
        if np.allclose(meta['Vt'], Vt_arr) and np.allclose(meta['Vb'], Vb_arr):
            print("Wczytano wykres B z cache.")
        else:
            raise ValueError("Inna siatka — przeliczam.")
    except Exception:
        print(f"Obliczam {len(Vt_arr)}×{len(Vb_arr)} = {len(Vt_arr)*len(Vb_arr)} pkt …")
        res = solver.solve_sweep(Vt_arr, Vb_arr)
        np.save(cache_file, res)
        np.savez(cache_meta, Vt=Vt_arr, Vb=Vb_arr)
        print("Cache zapisany.")

    # res[key] ma kształt (n_Vt, n_Vb) → transpose do (n_Vb, n_Vt) dla pcolormesh(x=Vt, y=Vb)
    Vt_2D, Vb_2D = np.meshgrid(Vt_arr, Vb_arr)
    n_tot_map = (res['n1'] + res['n2']).T
    U1_map    = res['U1'].T
    U2_map    = res['U2'].T

    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(float(np.max(np.abs(n_tot_map))), 1e-9)
    pcm = ax.pcolormesh(Vt_2D, Vb_2D, n_tot_map,
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r'$n_{tot} = n_1 + n_2$  ($10^{15}$ m$^{-2}$)', fontsize=10)

    cs1 = ax.contour(Vt_2D, Vb_2D, U1_map, levels=[0],
                     colors='black', linewidths=2.0, linestyles='-')
    cs2 = ax.contour(Vt_2D, Vb_2D, U2_map, levels=[0],
                     colors='limegreen', linewidths=2.0, linestyles='-')
    ax.clabel(cs1, fmt=r'$U_1=0$', fontsize=8, inline=True)
    ax.clabel(cs2, fmt=r'$U_2=0$', fontsize=8, inline=True)

    ax.set_xlabel(r'$V_t$ (V)', fontsize=12)
    ax.set_ylabel(r'$V_b$ (V)', fontsize=12)
    ax.set_title(
        r'Wykres B — Całkowita gęstość $n_{tot}(V_t, V_b)$'
        '\n'
        r'kontury: $U_1=0$ (czarny), $U_2=0$ (zielony)',
        fontsize=10,
    )
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.3)

    from matplotlib.lines import Line2D
    leg_elems = [
        Line2D([0], [0], color='k',         lw=2, label=r'$U_1 = 0$'),
        Line2D([0], [0], color='limegreen', lw=2, label=r'$U_2 = 0$'),
    ]
    ax.legend(handles=leg_elems, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/wykresB_mapa_2D_ntot.pdf")
    plt.savefig("plots/wykresB_mapa_2D_ntot.png", dpi=150)
    print("Zapisano: plots/wykresB_mapa_2D_ntot.pdf")
    # plt.show()
    return res


def zadanie_C_pasma_z_B():
    """
    Wykres C: Struktura pasmowa BLG (1×2 subplots).

    Lewy panel  (B = 0 T): gładkie, ciągłe pasma z przerwą energetyczną U=0.1 eV.
    Prawy panel (B = 4 T): płaskie poziomy Landaua w objętości + skośne stany
                           krawędziowe przecinające przerwę.

    Oś X: bezwymiarowy pęd krystaliczny k_y·a ∈ [-π, π] (pierwsza strefa Brillouina).
    Oś Y: energia E ∈ [-0.5, 0.5] eV.
    """
    print("\n" + "=" * 70)
    print("WYKRES C: Struktura pasmowa — B=0 T i B=4 T")
    print("=" * 70)

    ensure_plots_dir()

    s_f = 4.0
    # Czynnik przeskalowania k → pęd w jednostkach wewnętrznych Kwant (a.u.)
    _scale = s_f * A_GRAPHENE / l_scale    # = s_f · a_graphene [Bohr]

    # Pełna 1. strefa Brillouina: k·a ∈ [-π, π]
    k_BZ_dim   = np.linspace(-np.pi, np.pi, 500)          # bezwymiarowy k·a
    k_pts      = k_BZ_dim / _scale                         # k w jed. wewnętrznych

    U_gap = 0.1 / E_scale   # 100 meV przerwa

    common_kw = dict(
        L=100.0 / l_scale, W=50.0 / l_scale,
        s_f=s_f, t=T_INTRALAYER, gamma1=GAMMA1,
        U1=U_gap, V1=0.0,
    )

    # --- B = 0 T ---
    print("  Obliczam dyspersję B = 0 T …")
    params_B0 = BLGSystemParameters(**common_kw, B=0.0, name="wykresC_B0T")
    blg_B0    = make_blg_system(params_B0)
    E_B0      = calculate_dispersion(blg_B0, k_points=k_pts)

    # --- B = 4 T ---
    print("  Obliczam dyspersję B = 4 T …")
    params_B4 = BLGSystemParameters(**common_kw, B=4.0 / T_scale, name="wykresC_B4T")
    blg_B4    = make_blg_system(params_B4)
    E_B4      = calculate_dispersion(blg_B4, k_points=k_pts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        r'Wykres C — Struktura pasmowa BLG  ($s_f=4$, $W=50.0$ nm, $U=0.1$ eV)',
        fontsize=11,
    )

    E_lim = 0.5  # eV

    # Lewy panel: B = 0 T
    ax1.plot(k_BZ_dim, E_B0 * E_scale, 'b-', lw=0.4, alpha=0.7)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(-E_lim, E_lim)
    ax1.set_xlabel(r'$k_y\, a$', fontsize=12)
    ax1.set_ylabel(r'$E$ (eV)', fontsize=12)
    ax1.set_title(r'$B = 0$ T — gładkie pasma z przerwą', fontsize=10)
    ax1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax1.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    # Zaznacz krawędzie przerwy U/2
    ax1.axhline( U_gap * E_scale / 2, color='tomato',  lw=0.8, ls=':', alpha=0.8,
                label=r'$\pm U/2$')
    ax1.axhline(-U_gap * E_scale / 2, color='tomato',  lw=0.8, ls=':', alpha=0.8)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Prawy panel: B = 4 T
    ax2.plot(k_BZ_dim, E_B4 * E_scale, 'r-', lw=0.4, alpha=0.7)
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-E_lim, E_lim)
    ax2.set_xlabel(r'$k_y\, a$', fontsize=12)
    ax2.set_title(r'$B = 4$ T — poziomy Landaua + stany krawędziowe', fontsize=10)
    ax2.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax2.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax2.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax2.axhline( U_gap * E_scale / 2, color='tomato', lw=0.8, ls=':', alpha=0.8,
                label=r'$\pm U/2$')
    ax2.axhline(-U_gap * E_scale / 2, color='tomato', lw=0.8, ls=':', alpha=0.8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("plots/wykresC_pasma_z_B.pdf")
    plt.savefig("plots/wykresC_pasma_z_B.png", dpi=150)
    print("Zapisano: plots/wykresC_pasma_z_B.pdf")
    # plt.show()


def zadanie_D_mapa_G_Vb_B():
    """
    Wykres D: Mapa transportowa 2D — G(Vb, B).

    Oś X : Vb ∈ [0, 40] V (bramka dolna).
    Oś Y : B ∈ [0, 8] T.
    Kolor: przewodność G [2e²/h], obliczona z parametrami samouzgodnionymi
           przy stałym Vt = 0 V.

    Nałożona linia analityczna (biała przerywana):
        B(Vb) = ħ k_F / (e R_c),  R_c = 45 nm,  k_F = √(π |n_tot| × 10^15) m^-1.
    """
    print("\n" + "=" * 70)
    print("WYKRES D: Mapa G(Vb, B) z linią R_c = 45 nm")
    print("=" * 70)

    ensure_plots_dir()
    ensure_data_dir()

    # --- Siatka ---
    Vt_fixed  = 0.0
    Vb_arr    = np.arange(0.0, 41.0, 2.0)   # 21 pkt
    B_phys    = np.arange(0.0,  8.5, 0.5)   # 17 pkt
    n_B, n_Vb = len(B_phys), len(Vb_arr)
    print(f"Siatka: {n_B} × {n_Vb} = {n_B * n_Vb} pkt (Vt={Vt_fixed} V)")

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)

    base_params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0  / l_scale,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        name="wykresD",
    )

    cache_file = "data/wykresD_G_map.npy"
    cache_meta = "data/wykresD_G_map_meta.npz"
    G_map = None
    try:
        G_map = np.load(cache_file)
        meta  = np.load(cache_meta)
        if (G_map.shape != (n_B, n_Vb) or
                not np.allclose(meta['Vt'], [Vt_fixed]) or
                not np.allclose(meta['B'],  B_phys) or
                not np.allclose(meta['Vb'], Vb_arr)):
            raise ValueError("Niezgodna siatka.")
        print("Wczytano G_map z cache.")
    except Exception as exc:
        print(f"Cache nieważny ({exc}), obliczam …")
        G_map = conductance_map_G_B_Vb(
            base_params, B_phys, Vb_arr,
            Vt=Vt_fixed,
            cap_params=cap_params, phys_params=phys_params,
            verbose=True,
        )
        np.save(cache_file, G_map)
        np.savez(cache_meta, Vt=[Vt_fixed], B=B_phys, Vb=Vb_arr)
        print("Cache zapisany.")

    # --- Linia analityczna R_c = 45 nm ---
    hbar_SI = 1.0545718e-34   # J·s
    e_SI    = 1.602176634e-19  # C
    Rc_SI   = 45e-9            # m

    # Parametry samouzgodnione przy B=0 (tylko do wyznaczenia n_tot(Vb))
    solver = DoubleBLGSolver(cap_params, phys_params)
    sc     = solver.solve_1D_sweep(Vb_arr, Vt=Vt_fixed)
    n_tot_SI = np.abs(sc['n1'] + sc['n2']) * 1e15  # [m^-2]

    # k_F = sqrt(π * n_tot)  (BLG z degeneracją 4: n = k_F²/π)
    k_F_SI  = np.sqrt(np.pi * n_tot_SI)
    B_line  = hbar_SI * k_F_SI / (e_SI * Rc_SI)   # [T]

    # --- Wykres ---
    B_2D, Vb_2D = np.meshgrid(B_phys, Vb_arr)    # (n_Vb, n_B)

    fig, ax = plt.subplots(figsize=(9, 6))
    pcm = ax.pcolormesh(Vb_2D, B_2D, G_map.T,
                        cmap='viridis',
                        vmin=np.min(G_map), vmax=np.max(G_map),
                        shading='auto')
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r'$G$  (2$e^2/h$)', fontsize=11)

    # Linia R_c = 45 nm (tylko dla Vb gdzie B_line ≤ B_phys.max)
    mask = B_line <= B_phys.max()
    if mask.any():
        ax.plot(Vb_arr[mask], B_line[mask],
                'w--', lw=2.2, label=rf'$R_c = {int(Rc_SI*1e9)}$ nm')
        ax.legend(fontsize=10, loc='upper left',
                  framealpha=0.6, edgecolor='w', labelcolor='w',
                  facecolor='#333333')

    ax.set_xlabel(r'$V_b$ (V)', fontsize=12)
    ax.set_ylabel(r'$B$ (T)',   fontsize=12)
    ax.set_title(
        r'Wykres D — Mapa przewodności $G(V_b, B)$'
        '\n'
        rf'$V_t = {Vt_fixed:.0f}$ V,  parametry samouzgodnione'
        r',  linia: $R_c = ħk_F / eB = 45$ nm',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig("plots/wykresD_mapa_G_Vb_B.pdf")
    plt.savefig("plots/wykresD_mapa_G_Vb_B.png", dpi=150)
    print("Zapisano: plots/wykresD_mapa_G_Vb_B.pdf")
    # plt.show()
    return G_map


def zadanie_E_profil_potencjalu(
    Vt: float = 5.0,
    Vb: float = 5.0,
    B_T: float = 3.0,
    L_nm: float = 200.0,
    d_nm: float = 40.0,
):
    """
    Wykres E: Profil pola magnetycznego i potencjału wzdłuż złącza BLG (oś x).

    Subplot 1 — B(x): pole magnetyczne zmienia znak przy x=0 (skok prostokątny,
                efekt geometryczny złożenia próbki).
    Subplot 2 — pasma: profil V(x) i U(x) interpolowany przez tanh(x/d),
                zgodny z _junction_VU w BLG_system.py.

    Konwencja Kwant (BLG1 obrócony):
      U1_kwant = −U1_solver  →  warstwa interfejsowa jest górną warstwą x≥0.

    Args:
        Vt:    napięcie bramki górnej [V]
        Vb:    napięcie bramki dolnej [V]
        B_T:   pole magnetyczne [T]  (|B|; znak flipuje przy x=0)
        L_nm:  połowa długości układu [nm]
        d_nm:  szerokość przejścia tanh [nm]  (domyślnie = params.d = 50 nm)
    """
    print("\n" + "=" * 70)
    print(f"WYKRES E: Profil B i potencjału  Vt={Vt} V  Vb={Vb} V  B={B_T} T")
    print("=" * 70)

    ensure_plots_dir()

    cap_params  = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    solver = DoubleBLGSolver(cap_params, phys_params)

    sc = solver.solve(Vt, Vb)
    Vg1_eV =  sc.Vg1   # [eV] — prawa strona (x ≥ 0)
    U1_eV  = -sc.U1    # [eV] — negacja: BLG1 obrócony (top layer → bottom layer w Kwancie)
    Vg2_eV =  sc.Vg2   # [eV] — lewa strona (x < 0)
    U2_eV  =  sc.U2    # [eV]

    print(f"  BLG1 (x≥0):  Vg1 = {Vg1_eV:.4f} eV,  U1_kwant = {U1_eV:.4f} eV,  n1 = {sc.n1:.3f}×10¹⁵ m⁻²")
    print(f"  BLG2 (x<0):  Vg2 = {Vg2_eV:.4f} eV,  U2      = {U2_eV:.4f} eV,  n2 = {sc.n2:.3f}×10¹⁵ m⁻²")
    print(f"  Profil tanh: d = {d_nm:.1f} nm")

    # --- Oś x ---
    x_nm = np.linspace(-L_nm, L_nm, 4000)

    # --- Pole magnetyczne: płynne przejście tanh(x/d) ---
    B_x = B_T * np.tanh(x_nm / d_nm)

    # --- Potencjał i przerwa: profil tanh(x/d) ---
    tanh_x = np.tanh(x_nm / d_nm)
    V_x = (Vg1_eV + Vg2_eV) / 2 + (Vg1_eV - Vg2_eV) / 2 * tanh_x
    U_x = (U1_eV  + U2_eV)  / 2 + (U1_eV  - U2_eV)  / 2 * tanh_x

    low_layer   = V_x + U_x / 2
    upp_layer   = V_x - U_x / 2
    band_top    = V_x + np.abs(U_x) / 2
    band_bottom = V_x - np.abs(U_x) / 2

    # --- Rysunek: 2 subploty z wspólną osią x ---
    fig, (ax_B, ax_pot) = plt.subplots(
        2, 1, figsize=(9, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0.08},
    )
    fig.suptitle(
        rf"Wykres E — Pole magnetyczne i profil pasm w złączu BLG"
        "\n"
        rf"$V_t={Vt:.1f}$ V,  $V_b={Vb:.1f}$ V,  $B={B_T:.1f}$ T,  "
        rf"$d={d_nm:.0f}$ nm,  $C_t=C_m=2.55\times10^{{15}}$ m$^{{-2}}$V$^{{-1}}$",
        fontsize=10,
    )

    # --- Subplot 1: B(x) ---
    ax_B.step(x_nm, B_x, where='mid', color='darkorange', lw=2.0,
              label=rf'$B(x)$  (flip przy $x=0$)')
    ax_B.axhline(0,  color='gray', lw=0.5, ls='--', alpha=0.5)
    ax_B.axvline(0,  color='gray', lw=1.0, ls=':', alpha=0.7)
    ax_B.fill_between(x_nm, 0, B_x, alpha=0.15, color='darkorange')
    ax_B.set_ylabel(r'$B$ (T)', fontsize=11)
    ax_B.set_ylim(-B_T * 1.5, B_T * 1.5)
    ax_B.set_yticks([-B_T, 0, B_T])
    ax_B.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:+.1f}'))
    ax_B.legend(fontsize=8, loc='upper right')
    ax_B.grid(True, alpha=0.2)
    ax_B.text( L_nm * 0.55,  B_T * 0.6, rf'$+{B_T:.1f}$ T', ha='center',
               fontsize=9, color='darkorange')
    ax_B.text(-L_nm * 0.55, -B_T * 0.6, rf'$-{B_T:.1f}$ T', ha='center',
               fontsize=9, color='darkorange')

    # --- Subplot 2: pasma ---
    ax_pot.plot(x_nm, band_top,    color='navy',      lw=2.0,
                label=r'Pasmo przewodnictwa: $V+|U|/2$')
    ax_pot.plot(x_nm, band_bottom, color='firebrick', lw=2.0,
                label=r'Pasmo walencyjne: $V-|U|/2$')
    ax_pot.plot(x_nm, V_x,         color='black',     lw=1.5, ls='--',
                label=r'$V(x)$ (śr. potencjał)')
    ax_pot.fill_between(x_nm, band_bottom, band_top,
                        alpha=0.10, color='purple',
                        label=rf'Przerwa $|U(x)|$  [tanh, $d={d_nm:.0f}$ nm]')
    ax_pot.axvline(0, color='gray', lw=1.0, ls=':', alpha=0.7, label='Granica złącza')
    ax_pot.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

    # Adnotacje przerw energetycznych po obu stronach
    gap_r = abs(U1_eV)
    gap_l = abs(U2_eV)
    for xpos, gap, mid, col in [
        ( L_nm * 0.60, gap_r, Vg1_eV, 'navy'),
        (-L_nm * 0.60, gap_l, Vg2_eV, 'firebrick'),
    ]:
        lbl = rf'$|U|={gap*1e3:.1f}$ meV'
        ax_pot.annotate(lbl, xy=(xpos, mid), fontsize=8, color=col,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8))

    ax_pot.set_xlabel(r'$x$ (nm)', fontsize=11)
    ax_pot.set_ylabel(r'Energia (eV)', fontsize=11)
    ax_pot.legend(fontsize=8, loc='best')
    ax_pot.grid(True, alpha=0.25)
    ax_pot.set_xlim(-L_nm, L_nm)

    plt.tight_layout()
    fname = f"plots/wykresE_profil_potencjalu_Vt{Vt:.0f}_Vb{Vb:.0f}_B{B_T:.0f}"
    plt.savefig(fname + ".pdf")
    plt.savefig(fname + ".png", dpi=150)
    print(f"Zapisano: {fname}.pdf")
    # plt.show()

    return {
        'x_nm': x_nm,
        'B_x': B_x,
        'V_x': V_x,
        'U_x': U_x,
        'low_layer': low_layer,
        'upp_layer': upp_layer,
        'sc': sc,
    }


def main():
    """Główna funkcja uruchamiająca wszystkie zadania."""
    print("\n" + "=" * 70)
    print("SYMULACJE TRANSPORTU PRZEZ DWUWARSTWOWY GRAFEN (BILAYER)")
    print("=" * 70)
    
    # Utwórz katalogi
    ensure_data_dir()
    ensure_plots_dir()
    
    # Uruchom zadania
    # try:
    #     zadanie1_dyspersja()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 1: {e}")
    
    # try:
    #     zadanie2_dyspersja_z_przerwa()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 2: {e}")
    
    # try:
    #     zadanie4_parametry_samouzgodnione()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 4: {e}")

    # try:
    #     zadanie6_mapy_2D()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 6: {e}")
    
    # try:
    #     zadanie3_przewodnosc_vs_B()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 3: {e}")
    
    # try:
    #     zadanie5_przewodnosc_samouzgodniona()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 5: {e}")

    # try:
    #     zadanie6_mapy_2D()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 6: {e}")

    # try:
    #     zadanie8_mapa_G_Vb_B_Vt0()
    # except Exception as e:
    #     print(f"Błąd w zadaniu 8: {e}")

    # wykres_G_od_B_stale_Vb(Vb_target=15.0)
    try:
        zadanie9_prad_lokalny(Vt=0.0, Vb=15.0, B_T=6)
    except Exception as e:
        print(f"Błąd w zadaniu 9: {e}")

    # # --- Nowe wykresy A–D ---
    # try:
    #     zadanie_A_elektrostatyka_1D()
    # except Exception as e:
    #     print(f"Błąd w wykresie A: {e}")

    # try:
    #     zadanie_B_mapa_2D_ntot()
    # except Exception as e:
    #     print(f"Błąd w wykresie B: {e}")

    # try:
    #     zadanie_C_pasma_z_B()
    # except Exception as e:
    #     print(f"Błąd w wykresie C: {e}")

    # try:
    #     zadanie_D_mapa_G_Vb_B()
    # except Exception as e:
    #         print(f"Błąd w wykresie D: {e}")

    try:
        zadanie_E_profil_potencjalu(Vt=0.0, Vb=15.0, B_T=6)
    except Exception as e:
        print(f"Błąd w wykresie E: {e}")

    print("\n" + "=" * 70)
    print("ZAKOŃCZONO SYMULACJE")
    print("=" * 70)


if __name__ == "__main__":
    main()
