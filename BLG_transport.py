"""
Moduł do obliczeń transportowych dla dwuwarstwowego grafenu (bilayer graphene).

Zawiera funkcje do:
- Obliczania transmisji i przewodności
- Relacji dyspersji
- Zależności od pola magnetycznego
- Zależności od napięcia bramkowego
"""

import numpy as np
import kwant
from kwant.physics import Bands
from typing import Optional, Tuple
import pickle
from pathlib import Path

from Constants import l_scale, E_scale, T_scale, h, e, k_arr, B_arr
from BLG_system import (
    BLGSystemParameters, BilayerGrapheneSystem,
    make_blg_system, make_blg_unit_cell, make_blg_parametric_system,
    T_INTRALAYER, GAMMA1, A_GRAPHENE
)
from BLG_parameters import DoubleBLGSolver, CapacitanceParameters, BLGPhysicsParameters


# ============================================================================
# FUNKCJE ZAPISU DANYCH
# ============================================================================

def ensure_data_dir(path: str = "data"):
    """Utwórz katalog danych, jeśli nie istnieje."""
    Path(path).mkdir(exist_ok=True)


def ensure_plots_dir(path: str = "plots"):
    """Utwórz katalog z wykresami, jeśli nie istnieje."""
    Path(path).mkdir(exist_ok=True)


def save_parameters(params: BLGSystemParameters):
    """
    Zapisz parametry systemu do pliku pickle.
    
    Args:
        params: Parametry systemu BLG
    """
    ensure_data_dir()
    data = {
        'L': params.L * l_scale,  # konwersja z powrotem do nm
        'W': params.W * l_scale,
        't': params.t * E_scale,   # konwersja do eV
        'gamma1': params.gamma1 * E_scale,
        's_f': params.s_f,
        'B': params.B * T_scale,   # konwersja do T
        'V1': params.V1 * E_scale,
        'U1': params.U1 * E_scale,
        'd': params.d * l_scale,
        'name': params.name
    }
    with open(f"data/{params.name}_params.pkl", "wb") as f:
        pickle.dump(data, f)


# ============================================================================
# OBLICZENIA STRUKTURY PASMOWEJ
# ============================================================================

def calculate_dispersion(blg: BilayerGrapheneSystem, 
                        lead_idx: int = 1,
                        k_points: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Oblicz relację dyspersji w przewodniku.
    
    Args:
        blg: System BLG
        lead_idx: Indeks przewodnika (0 = lewy, 1 = prawy)
        k_points: Tablica wektorów falowych (opcjonalna)
        
    Returns:
        Tablica energii dla każdego k
    """
    bands = Bands(blg.system.leads[lead_idx])
    
    if k_points is None:
        k_points = k_arr
    
    # Skalowanie pędów
    momenta = k_points * (blg.params.s_f * A_GRAPHENE / l_scale)
    
    energies = np.array([bands(k) for k in momenta])

    return energies


def calculate_band_structure(params: BLGSystemParameters,
                            k_path: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Oblicz strukturę pasmową z komórki elementarnej.
    
    Args:
        params: Parametry systemu
        k_path: Ścieżka w przestrzeni k
        
    Returns:
        Tablica energii
    """
    unit_cell = make_blg_unit_cell(params)
    bands = Bands(unit_cell.finalized())
    
    if k_path is None:
        k_path = np.linspace(-0.5, 0.5, 200)
    
    momenta = k_path * (params.s_f * A_GRAPHENE / l_scale)
    energies = np.array([bands(k) for k in momenta])
    
    return energies


# ============================================================================
# OBLICZENIA TRANSMISJI I PRZEWODNOŚCI
# ============================================================================

def calculate_transmission(blg: BilayerGrapheneSystem, 
                          energy: float) -> float:
    """
    Oblicz transmisję przy danej energii.
    
    Args:
        blg: System BLG
        energy: Energia w jednostkach atomowych
        
    Returns:
        Transmisja T(E)
    """
    smatrix = kwant.smatrix(blg.system, energy=energy, 
                           in_leads=(0, 1), out_leads=(0, 1))
    return smatrix.transmission(1, 0)


def calculate_conductance(blg: BilayerGrapheneSystem,
                         energy: float = 0.0) -> float:
    """
    Oblicz przewodność w jednostkach 2e²/h.
    
    Args:
        blg: System BLG
        energy: Energia Fermiego
        
    Returns:
        Przewodność G w jednostkach 2e²/h
    """
    return calculate_transmission(blg, energy)


def transmission_vs_energy(blg: BilayerGrapheneSystem,
                          E_arr: np.ndarray) -> np.ndarray:
    """
    Oblicz transmisję jako funkcję energii.
    
    Args:
        blg: System BLG
        E_arr: Tablica energii [jednostki atomowe]
        
    Returns:
        Tablica transmisji
    """
    T_arr = np.array([calculate_transmission(blg, E) for E in E_arr])
    
    ensure_data_dir()
    np.save(f"data/{blg.params.name}_T_vs_E.npy", T_arr)
    np.save(f"data/{blg.params.name}_E_arr.npy", E_arr)
    
    return T_arr


def conductance_vs_magnetic_field(base_params: BLGSystemParameters,
                                  B_values: np.ndarray,
                                  energy: float = 0.0) -> np.ndarray:
    """
    Oblicz przewodność jako funkcję pola magnetycznego.
    
    Args:
        base_params: Bazowe parametry systemu
        B_values: Tablica wartości pola B [jednostki atomowe]
        energy: Energia Fermiego
        
    Returns:
        Tablica przewodności
    """
    # Buduj system raz i parametryzuj B w wywołaniu smatrix
    p_par = BLGSystemParameters(
        L=base_params.L,
        W=base_params.W,
        t=base_params.t,
        gamma1=base_params.gamma1,
        s_f=base_params.s_f,
        B=base_params.B,
        d=base_params.d,
        name=f"{base_params.name}_par"
    )
    fsyst = make_blg_parametric_system(p_par).system

    G_arr = []

    for B in B_values:
        sm = kwant.smatrix(
            fsyst,
            energy=energy,
            params=dict(
                B=B,
                V1=base_params.V1,
                U1=base_params.U1,
                V2=base_params.V2,
                U2=base_params.U2,
            ),
            in_leads=(0, 1),
            out_leads=(0, 1)
        )
        G_arr.append(sm.transmission(1, 0))
        
    G_arr = np.array(G_arr)
    
    ensure_data_dir()
    np.save(f"data/{base_params.name}_G_vs_B.npy", G_arr)
    np.save(f"data/{base_params.name}_B_arr.npy", B_values)
    
    return G_arr


def conductance_vs_gate_voltage(base_params: BLGSystemParameters,
                                Vg_values: np.ndarray,
                                energy: float = 0.0) -> np.ndarray:
    """
    Oblicz przewodność jako funkcję napięcia bramkowego.
    
    Napięcie bramkowe modyfikuje potencjał V1.
    
    Args:
        base_params: Bazowe parametry systemu
        Vg_values: Tablica napięć bramkowych [V]
        energy: Energia Fermiego
        
    Returns:
        Tablica przewodności
    """
    G_arr = []
    
    for Vg in Vg_values:
        # Konwersja napięcia do jednostek atomowych
        V1 = Vg / E_scale
        
        params = BLGSystemParameters(
            L=base_params.L,
            W=base_params.W,
            t=base_params.t,
            gamma1=base_params.gamma1,
            s_f=base_params.s_f,
            B=base_params.B,
            V1=V1,
            U1=base_params.U1,
            d=base_params.d,
            name=base_params.name
        )
        blg = make_blg_system(params)
        G = calculate_conductance(blg, energy)
        G_arr.append(G)
        
    G_arr = np.array(G_arr)
    
    ensure_data_dir()
    np.save(f"data/{base_params.name}_G_vs_Vg.npy", G_arr)
    np.save(f"data/{base_params.name}_Vg_arr.npy", Vg_values)
    
    return G_arr


def conductance_vs_gap(base_params: BLGSystemParameters,
                       U_values: np.ndarray,
                       energy: float = 0.0) -> np.ndarray:
    """
    Oblicz przewodność jako funkcję przerwy energetycznej U.
    
    Args:
        base_params: Bazowe parametry systemu
        U_values: Tablica wartości przerwy U [eV]
        energy: Energia Fermiego
        
    Returns:
        Tablica przewodności
    """
    G_arr = []
    
    for U in U_values:
        U1 = U / E_scale  # konwersja do jednostek atomowych
        
        params = BLGSystemParameters(
            L=base_params.L,
            W=base_params.W,
            t=base_params.t,
            gamma1=base_params.gamma1,
            s_f=base_params.s_f,
            B=base_params.B,
            V1=base_params.V1,
            U1=U1,
            d=base_params.d,
            name=base_params.name
        )
        blg = make_blg_system(params)
        G = calculate_conductance(blg, energy)
        G_arr.append(G)
        
    G_arr = np.array(G_arr)
    
    ensure_data_dir()
    np.save(f"data/{base_params.name}_G_vs_U.npy", G_arr)
    np.save(f"data/{base_params.name}_U_arr.npy", U_values)
    
    return G_arr


# ============================================================================
# OBLICZENIA Z SAMOUZGODNIONYMI PARAMETRAMI
# ============================================================================

def conductance_with_selfconsistent_params(
    base_params: BLGSystemParameters,
    Vb_values: np.ndarray,
    Vt: float = 0.0,
    bilayer_index: int = 1,
    cap_params: Optional[CapacitanceParameters] = None,
    phys_params: Optional[BLGPhysicsParameters] = None
) -> Tuple[np.ndarray, dict]:
    """
    Oblicz przewodność z samouzgodnionymi parametrami.
    
    Używa solvera DoubleBLGSolver do wyznaczenia U i V dla każdego Vb,
    następnie oblicza przewodność przez wybrany bilayer.
    
    Args:
        base_params: Bazowe parametry systemu
        Vb_values: Tablica napięć bramki dolnej [V]
        Vt: Napięcie bramki górnej [V]
        bilayer_index: Który bilayer symulować (1 lub 2)
        cap_params: Parametry pojemnościowe
        phys_params: Parametry fizyczne BLG
        
    Returns:
        Tuple: (tablica przewodności, słownik z parametrami samouzgodnionymi)
    """
    # Utwórz solver
    solver = DoubleBLGSolver(cap_params, phys_params)
    
    # Rozwiąż równania samouzgodnione
    sc_results = solver.solve_1D_sweep(Vb_values, Vt)
    
    # Buduj system RAZ (B jest stały = base_params.B) — reużyj dla wszystkich Vb
    p_b = BLGSystemParameters(
        L=base_params.L, W=base_params.W,
        t=base_params.t, gamma1=base_params.gamma1,
        s_f=base_params.s_f, B=base_params.B, d=base_params.d,
        name=f"{base_params.name}_par"
    )
    fsyst = make_blg_parametric_system(p_b).system

    # Oblicz przewodność
    G_arr = []

    for i, Vb in enumerate(Vb_values):
        # Prawa strona złącza: BLG1 (górny, z bramką górną)
        V1 = sc_results['Vg1'][i] / E_scale
        U1 = -sc_results['U1'][i] / E_scale
        # Lewa strona złącza: BLG2 (dolny, z bramką dolną)
        V2 = sc_results['Vg2'][i] / E_scale
        U2 = sc_results['U2'][i] / E_scale
        energy = 0.0
        sm = kwant.smatrix(fsyst, energy=energy,
                           params=dict(B=base_params.B, V1=V1, U1=U1, V2=V2, U2=U2))
        G_arr.append(sm.transmission(1, 0))
        
    G_arr = np.array(G_arr)
    
    ensure_data_dir()
    np.save(f"data/{base_params.name}_G_vs_Vb_selfconsistent.npy", G_arr)
    
    return G_arr, sc_results


# ============================================================================
# MAPA PRZEWODNOŚCI G(B, Vb)
# ============================================================================

def conductance_map_G_B_Vb(
    base_params: BLGSystemParameters,
    B_phys_arr: np.ndarray,
    Vb_arr: np.ndarray,
    Vt: float = 0.0,
    cap_params: Optional[CapacitanceParameters] = None,
    phys_params: Optional[BLGPhysicsParameters] = None,
    uniform_B: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Oblicz mapę przewodności G(B, Vb) dla stałego Vt.

    Parametry samouzgodnione (V1, U1, V2, U2) wyznaczone przy B=0,
    następnie dla każdego B obliczana jest przewodność Kwanta.
    Profil potencjału V/U jest zachowany. Przy `uniform_B=True` pole B
    jest stałe w obszarze centralnym i w obu leadach, zamiast profilu
    `B*tanh(x/d)`.

    Args:
        base_params:  Bazowe parametry geometrii systemu
        B_phys_arr:   Tablica wartości B [T]
        Vb_arr:       Tablica napięć bramki dolnej [V]
        Vt:           Napięcie bramki górnej [V]
        cap_params:   Parametry pojemnościowe (opcjonalne)
        phys_params:  Parametry fizyczne BLG (opcjonalne)
        verbose:      Czy drukować postęp

    Returns:
        G_map: ndarray shape (n_B, n_Vb)
    """
    solver = DoubleBLGSolver(cap_params, phys_params)
    sc = solver.solve_1D_sweep(Vb_arr, Vt)

    n_B, n_Vb = len(B_phys_arr), len(Vb_arr)
    G_map = np.zeros((n_B, n_Vb))

    # Buduj system tylko raz; B i V/U przekazuj jako params do smatrix
    p_par = BLGSystemParameters(
        L=base_params.L,
        W=base_params.W,
        t=base_params.t,
        gamma1=base_params.gamma1,
        s_f=base_params.s_f,
        B=base_params.B,
        d=base_params.d,
        name=f"{base_params.name}_par_G_B_Vb"
    )
    fsyst = make_blg_parametric_system(p_par, uniform_B=uniform_B).system

    for i_b, B_T in enumerate(B_phys_arr):
        B_au = B_T / T_scale
        for i_vb in range(n_Vb):
            if verbose:
                print(f"    B = {B_T:.1f} T  ({i_b+1}/{n_B}) i_vb = {i_vb+1}/{n_Vb}",flush=True)
            V1 = sc['Vg1'][i_vb] / E_scale
            U1 = -sc['U1'][i_vb] / E_scale
            V2 = sc['Vg2'][i_vb] / E_scale
            U2 = sc['U2'][i_vb]  / E_scale
            energy = 0.0
            sm = kwant.smatrix(fsyst, energy=energy,
                               params=dict(B=B_au, V1=V1, U1=U1, V2=V2, U2=U2))
            G_map[i_b, i_vb] = sm.transmission(1, 0)

    return G_map


# ============================================================================
# OBLICZENIA GĘSTOŚCI PRĄDU
# ============================================================================


def calculate_total_bond_current(blg: BilayerGrapheneSystem, energy: float) -> list:
    """
    Oblicz prąd wiązań dla każdego z modów z ołowiu 0.

    Args:
        blg:    System BLG (po make_blg_system)
        energy: Energia Fermiego [j.a.]

    Returns:
        J: lista tablic (lub pusta lista) — prąd każdego modu osobno
    """
    current_op = kwant.operator.Current(blg.system)

    # Pobieramy funkcję falową dla danej energii
    try:
        wf = kwant.wave_function(blg.system, energy)
        modes = wf(1)  # Mody propagujące się z ołowiu 1 (prawy → lewy) wh(1)
    except Exception as e:
        print(f"  [Błąd wave_function] Nie udało się pobrać modów: {e}")
        return []
    print(f"  Liczba propagujących modów: {len(modes)}", flush=True)
    J = []
    for i_mod, psi in enumerate(modes):
        print(f"    Obliczam prąd dla modu {i_mod+1}/{len(modes)} …", flush=True)
        J_current = current_op(psi)
        J.append(J_current)

    if not J:
        return []
    return J


def calculate_current_density(blg: BilayerGrapheneSystem,
                              energy: float,
                              lead_idx: int = 0,
                              mode_idx: int = 0) -> np.ndarray:
    """
    Oblicz rozkład gęstości prądu w systemie.
    
    Args:
        blg: System BLG
        energy: Energia
        lead_idx: Indeks przewodnika źródłowego
        mode_idx: Indeks modu w przewodniku
        
    Returns:
        Tablica gęstości prądu na wiązaniach
    """
    current_op = kwant.operator.Current(blg.system).bind()
    wf = kwant.wave_function(blg.system, energy)
    psi = wf(lead_idx)
    
    if mode_idx >= len(psi):
        raise ValueError(f"Mod {mode_idx} nie istnieje. Dostępne: {len(psi)}")
    
    current = current_op(psi[mode_idx])
    
    ensure_data_dir()
    np.save(f"data/{blg.params.name}_current.npy", current)
    
    return current


def calculate_ldos(blg: BilayerGrapheneSystem,
                   energy: float) -> np.ndarray:
    """
    Oblicz lokalną gęstość stanów (LDOS).
    
    Args:
        blg: System BLG
        energy: Energia
        
    Returns:
        Tablica LDOS na węzłach
    """
    ldos_op = kwant.operator.Density(blg.system).bind()
    wf = kwant.wave_function(blg.system, energy)
    
    ldos = np.zeros(blg.system.graph.num_nodes)
    
    for lead_idx in range(len(blg.system.leads)):
        psi_modes = wf(lead_idx)
        for psi in psi_modes:
            ldos += ldos_op(psi)
    
    ensure_data_dir()
    np.save(f"data/{blg.params.name}_ldos.npy", ldos)
    
    return ldos


# ============================================================================
# PRZYKŁADOWE OBLICZENIA
# ============================================================================

def example_dispersion():
    """Przykład: oblicz relację dyspersji dla różnych U."""
    print("=" * 60)
    print("PRZYKŁAD: Relacja dyspersji BLG dla różnych przerw U")
    print("=" * 60)
    
    ensure_plots_dir()
    
    U_values = [0.0, 0.05, 0.1, 0.2]  # eV
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(U_values), figsize=(12, 3))
    
    for i, U in enumerate(U_values):
        params = BLGSystemParameters(
            L=30.0 / l_scale,
            W=25.0 / l_scale,
            s_f=8.0,
            U1=U / E_scale,
            name=f"dispersion_U{U}"
        )
        
        blg = make_blg_system(params)
        E = calculate_dispersion(blg)
        
        axes[i].plot(k_arr, E * E_scale)
        axes[i].set_xlabel("k (1/nm)")
        axes[i].set_ylabel("E (eV)")
        axes[i].set_ylim(-0.5, 0.5)
        axes[i].set_title(f"U = {U} eV")
        axes[i].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/BLG_dispersion_vs_U.png", dpi=150)
    print("Zapisano: plots/BLG_dispersion_vs_U.png")
    plt.show()


def example_conductance_vs_B():
    """Przykład: przewodność vs pole magnetyczne."""
    print("=" * 60)
    print("PRZYKŁAD: Przewodność BLG vs pole magnetyczne")
    print("=" * 60)
    
    ensure_plots_dir()
    
    params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0 / l_scale,
        s_f=16.0,
        U1=0.1 / E_scale,
        name="G_vs_B"
    )
    
    B_values = np.linspace(0, 3.0, 31) / T_scale
    G = conductance_vs_magnetic_field(params, B_values)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(B_values * T_scale, G, 'o-')
    ax.set_xlabel("B (T)")
    ax.set_ylabel("G (2e²/h)")
    ax.set_title(f"Przewodność BLG\nU = {params.U1 * E_scale:.2f} eV")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/BLG_G_vs_B.png", dpi=150)
    print("Zapisano: plots/BLG_G_vs_B.png")
    plt.show()


def example_conductance_vs_Vg():
    """Przykład: przewodność vs napięcie bramkowe z parametrami samouzgodnionymi."""
    print("=" * 60)
    print("PRZYKŁAD: Przewodność BLG vs Vb (parametry samouzgodnione)")
    print("=" * 60)
    
    ensure_plots_dir()
    
    params = BLGSystemParameters(
        L=100.0 / l_scale,
        W=50.0 / l_scale,
        s_f=16.0,
        name="G_vs_Vb_sc"
    )
    
    Vb_values = np.linspace(-30, 30, 31)  # V
    G, sc_results = conductance_with_selfconsistent_params(params, Vb_values)
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Przewodność
    axes[0, 0].plot(Vb_values, G, 'o-')
    axes[0, 0].set_xlabel("Vb (V)")
    axes[0, 0].set_ylabel("G (2e²/h)")
    axes[0, 0].set_title("Przewodność")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Przerwa U
    axes[0, 1].plot(Vb_values, sc_results['U1'], 'b-', label='U₁')
    axes[0, 1].plot(Vb_values, sc_results['U2'], 'r--', label='U₂')
    axes[0, 1].set_xlabel("Vb (V)")
    axes[0, 1].set_ylabel("U (eV)")
    axes[0, 1].set_title("Przerwa energetyczna")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gęstość
    axes[1, 0].plot(Vb_values, sc_results['n1'], 'b-', label='n₁')
    axes[1, 0].plot(Vb_values, sc_results['n2'], 'r--', label='n₂')
    axes[1, 0].set_xlabel("Vb (V)")
    axes[1, 0].set_ylabel("n (10¹⁵ m⁻²)")
    axes[1, 0].set_title("Gęstość nośników")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Potencjał
    axes[1, 1].plot(Vb_values, sc_results['Vg1'], 'b-', label='Vg₁')
    axes[1, 1].plot(Vb_values, sc_results['Vg2'], 'r--', label='Vg₂')
    axes[1, 1].set_xlabel("Vb (V)")
    axes[1, 1].set_ylabel("Vg (V)")
    axes[1, 1].set_title("Potencjał bramkowy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/BLG_G_vs_Vb_selfconsistent.png", dpi=150)
    print("Zapisano: plots/BLG_G_vs_Vb_selfconsistent.png")
    plt.show()


if __name__ == "__main__":
    print("Moduł transportowy dla dwuwarstwowego grafenu")
    print("=" * 60)
    
    # Uruchom przykłady
    example_dispersion()
    # example_conductance_vs_B()
    # example_conductance_vs_Vg()
