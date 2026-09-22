import numpy as np
import kwant
from dataclasses import dataclass, field

# =============================================================================
# STAŁE JEDNOSTKOWE I SKALUJĄCE — przeliczniki jednostek atomowych
# =============================================================================
l_scale = 0.0529177249  # długość w nm (promień Bohra a_0)
E_scale = 27.211  # energia w eV (Hartree)
t_scale = 0.02418884  # czas w fs 
T_scale = 2.35051756758e5  # natężenie pola w T (teslach)

# Stałe fizyczne w jednostkach naturalnych (atomowych)
h = 1.0  # stała Plancka (ħ = 1)
e = 1.0  # ładunek elementarny
k_arr = np.linspace(-0.3, 0.3, 200)
B_arr = np.arange(1.0, 3.05, 0.05) / T_scale

# =============================================================================
# STAŁE DLA GRAFENU I DWUWARSTWOWEGO GRAFENU (BILAYER)
# =============================================================================

# Stała sieci grafenu
A_GRAPHENE_NM = 0.246  # nm - stała sieci (odległość między komórkami)
A_CC_NM = 0.142  # nm - odległość C-C w grafenie

# Konwersja do jednostek atomowych
A_GRAPHENE_AU = A_GRAPHENE_NM / l_scale  # stała sieci w jednostkach atomowych

# Energie przeskoku (w eV, do konwersji użyj /E_scale)
T_HOPPING_EV = -3.0  # eV - przeskok wewnątrzwarstwowy (t)
GAMMA1_EV = 0.39  # eV - przeskok międzywarstwowy (γ₁) w bilayer
GAMMA3_EV = 0.315  # eV - przeskok γ₃ (trigonal warping) - opcjonalny
GAMMA4_EV = 0.044  # eV - przeskok γ₄ - opcjonalny

# Odległość między warstwami
D_INTERLAYER_NM = 0.335  # nm - odległość między warstwami grafenu
D_BLG_INTRALAYER_NM = 0.12  # nm - odległość między warstwami w bilayerze (do pojemności)

# Prędkość Fermiego
V_FERMI = 1.0e6  # m/s - prędkość Fermiego w grafenie
HV_FERMI_EV_NM = 0.658  # eV·nm - ħv_F

# =============================================================================
# PARAMETRY POJEMNOŚCIOWE DLA PODWÓJNEGO BILAYERA (w jednostkach 10^15 m^-2 V^-1)
# =============================================================================
# Wzór: C/e = ε₀ * ε_r / (e * d)

# Pojemność między bilayerami (hBN, ε=3.7, d=80nm)
C_INTERBILAYER = 2.55  # [10^15 m^-2 V^-1]

# Pojemność bramki dolnej (SiO2, ε=3.9, d=330nm)
C_BACKGATE = 0.653  # [10^15 m^-2 V^-1]

# Pojemność między warstwami w bilayerze (próżnia/grafen, ε≈1, d=0.12nm)
# Uwaga: w eksperymencie może być ~230, teoretycznie ~460
C_INTRALAYER = 230.0  # [10^15 m^-2 V^-1]

# Charakterystyczna gęstość nośników dla BLG
N_CHARACTERISTIC = 118.57  # [10^15 m^-2]


@dataclass
class SystemParameters:
    """
    Data class to hold parameters of the quantum system.

    Attributes:
        dx (float): Lattice constant.
        L (float): Length of the system.
        W (float): Width of the system.
        m_eff (float): Effective mass of the electron.
        V_0 (float): Amplitude of the potential.
        x_0 (float): X position of the potential center.
        y_0 (float): Y position of the potential center.
        sigma (float): Width of the potential (standard deviation).
        B_z (float): Magnetic field strength perpendicular to the system.
        R1 (float): Inner radius of the ring (only for round systems).
        R2 (float): Outer radius of the ring (only for round systems).
        name (str): Name identifier used for saving output data.
    """

    dx: float = 0.0
    L: float = 0.0
    W: float = 0.0
    Vp: float = 0.0
    d: float = 1.0
    B: float = 0.0
    s_f: float = 0.0
    t: float = 0.0
    name: str = "_"


class FiniteSystem:
    """
    Wrapper for Kwant's FiniteSystem with additional parameters.

    Args:
        system (kwant.builder.FiniteSystem): Finalized Kwant system.
        sp (SystemParameters): Parameters used to construct the system.
    """

    def __init__(self, system: kwant.builder.FiniteSystem, sp: SystemParameters):
        self._system = system
        self.param = sp

    def __getattr__(self, name):
        """Delegate attribute access to the underlying Kwant system."""
        return getattr(self._system, name)

