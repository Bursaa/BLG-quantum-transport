"""
Moduł do tworzenia systemów kwantowych dla dwuwarstwowego grafenu (bilayer graphene)
z wykorzystaniem biblioteki Kwant.

Implementacja bazuje na modelu tight-binding z:
- Dwiema podsiećami w każdej warstwie (A, B)  
- Przeskoki wewnątrzwarstwowe t ≈ -3.0 eV
- Przeskok międzywarstwowy γ₁ ≈ 0.39 eV (między B dolnej i A górnej warstwy)
- Skalowanie siatki (scalable tight binding model)

Jednostki: atomowe (ħ = e = m_e = 1)
"""

import numpy as np
import kwant
from dataclasses import dataclass
from typing import Optional, Tuple
from Constants import l_scale, E_scale, h, e


# ============================================================================
# STAŁE DLA DWUWARSTWOWEGO GRAFENU
# ============================================================================

# Stała sieci grafenu
A_GRAPHENE = 0.246  # nm - stała sieci grafenu (odległość między komórkami)
A_CC = 0.142  # nm - odległość C-C w grafenie

# Energie przeskoku
T_INTRALAYER = -3.0 / E_scale  # przeskok wewnątrzwarstwowy w jednostkach atomowych
GAMMA1 = 0.39 / E_scale  # przeskok międzywarstwowy γ₁ w jednostkach atomowych

# Odległość między warstwami
D_INTERLAYER = 0.335  # nm - odległość między warstwami grafenu


@dataclass
class BLGSystemParameters:
    """
    Parametry systemu dwuwarstwowego grafenu.
    
    Attributes:
        L: Połowa długości systemu w kierunku x [jednostki atomowe]
        W: Połowa szerokości systemu w kierunku y [jednostki atomowe]
        t: Przeskok wewnątrzwarstwowy [jednostki atomowe]
        gamma1: Przeskok międzywarstwowy γ₁ [jednostki atomowe]
        s_f: Współczynnik skalowania siatki (scalable tight binding)
        B: Pole magnetyczne prostopadłe do płaszczyzny [jednostki atomowe]
        V1: Potencjał na bilayerze (średnia) [jednostki atomowe]
        U1: Przerwa energetyczna (różnica między warstwami) [jednostki atomowe]
        V2: Potencjał na drugim bilayerze (jeśli używany)
        U2: Przerwa energetyczna drugiego bilayera
        d: Szerokość profilu potencjału (dla funkcji tanh) [jednostki atomowe]
        name: Nazwa systemu do zapisu danych
    """
    L: float = 200.0 / l_scale      # połowa długości
    W: float = 80.0 / l_scale       # połowa szerokości  
    t: float = T_INTRALAYER         # przeskok wewnątrzwarstwowy
    gamma1: float = GAMMA1          # przeskok międzywarstwowy
    s_f: float = 16.0               # współczynnik skalowania
    B: float = 0.0                  # pole magnetyczne
    V1: float = 0.0                 # potencjał bilayera
    U1: float = 0.0                 # przerwa energetyczna
    V2: float = 0.0                 # potencjał drugiego bilayera (nieużywany dla pojedynczego)
    U2: float = 0.0                 # przerwa drugiego bilayera
    d: float = 5.0 / l_scale       # szerokość profilu potencjału
    name: str = "BLG_system"
    
    def __post_init__(self):
        """Walidacja parametrów."""
        if self.s_f <= 0:
            raise ValueError("Współczynnik skalowania s_f musi być dodatni")
        if self.L <= 0 or self.W <= 0:
            raise ValueError("Wymiary L i W muszą być dodatnie")
        if self.d <= 0:
            raise ValueError("Szerokość profilu d musi być dodatnia")

class BilayerGrapheneSystem:
    """
    Klasa do tworzenia i zarządzania systemem dwuwarstwowego grafenu w Kwant.
    
    System składa się z dwóch warstw grafenu:
    - Warstwa dolna (lower): podsiatki low_a, low_b
    - Warstwa górna (upper): podsiatki upp_a, upp_b
    
    Przeskoki:
    - t: między najbliższymi sąsiadami w tej samej warstwie (A-B)
    - γ₁: między B dolnej a A górnej warstwy (dimer sites)
    """

    def __init__(self, params: BLGSystemParameters):
        """
        Inicjalizacja systemu.
        
        Args:
            params: Parametry systemu BLG
        """
        self.params = params
        self._system: Optional[kwant.builder.FiniteSystem] = None
        self._use_profile: bool = False

        # Skalowana stała sieci
        self.a = A_GRAPHENE / l_scale * params.s_f

        # Skalowany przeskok wewnątrzwarstwowy
        self.t_scaled = params.t / params.s_f

        # Przeskok międzywarstwowy NIE jest skalowany
        self.gamma1 = params.gamma1

        # Tworzenie siatek
        self._create_lattices()

    def _create_lattices(self):
        """Utwórz siatki dla obu warstw grafenu - orientacja armchair wzdłuż x.

        Zgodna z kodem referencyjnym (make_graphene_system):
            a_cc = a/√3   (długość wiązania C-C)
            v1 = (0, a)                     ← wzdłuż y
            v2 = (√3/2·a, a/2)             ← skośny
        Atom A w (0, 0), atom B w (a_cc, 0) — wiązanie wzdłuż x.
        Translacja leadu: 2·v2 − v1 = (√3·a, 0) wzdłuż x.

        Stos Bernala (AB): low_b i upp_a są w tej samej pozycji (a_cc, 0),
        low_a i upp_b są w (0, 0) — pary dimer/non-dimer.
        """
        a = self.a
        a_cc = a / np.sqrt(3.0)   # długość wiązania C-C przeskalowana

        # Wektory Bravais'a - v1 wzdłuż y, v2 skośny
        bravais_vectors = [
            (0, a),
            (np.sqrt(3.0) / 2 * a, 0.5 * a),
        ]

        # Dolna warstwa: A na (0,0), B na (a_cc, 0)
        lower_positions = [
            (0, 0),     # A - dolna (non-dimer)
            (a_cc, 0),  # B - dolna (dimer z A górną - to samo miejsce)
        ]

        # Górna warstwa: A na (a_cc, 0) — nad B dolną (dimer), B na (0, 0)
        upper_positions = [
            (a_cc, 0),  # A - górna (dimer z B dolną - to samo miejsce)
            (2.0 * a_cc, 0),  # B - górna (non-dimer)
        ]

        # Utworzenie siatek
        self.lower_lattice = kwant.lattice.general(
            bravais_vectors, lower_positions, norbs=1
        )
        self.low_a, self.low_b = self.lower_lattice.sublattices

        self.upper_lattice = kwant.lattice.general(
            bravais_vectors, upper_positions, norbs=1
        )
        self.upp_a, self.upp_b = self.upper_lattice.sublattices

    def _rect_shape(self, pos: Tuple[float, float]) -> bool:
        """Kształt prostokątny dla centralnego regionu."""
        x, y = pos
        return -self.params.L <= x <= self.params.L and -self.params.W <= y <= self.params.W

    def _lead_shape(self, pos: Tuple[float, float]) -> bool:
        """Kształt przewodników (leads)."""
        x, y = pos
        return -self.params.W < y < self.params.W

    def _onsite_lower_a(self, site) -> float:
        """Energia on-site dla atomów A w dolnej warstwie."""
        x, y = site.pos
        # Dolna warstwa: Vg + U/2  (U1 to przerwa miedzywarstwowa)
        return self.params.V1 + self.params.U1 / 2

    def _onsite_lower_b(self, site) -> float:
        """Energia on-site dla atomów B w dolnej warstwie."""
        x, y = site.pos
        # Dolna warstwa: oba podwęzły mają tę samą energię warstwową
        return self.params.V1 + self.params.U1 / 2

    def _onsite_upper_a(self, site) -> float:
        """Energia on-site dla atomów A w górnej warstwie."""
        x, y = site.pos
        # Górna warstwa: Vg - U/2  (przeciwny znak przerwy miedzywarstowej)
        return self.params.V1 - self.params.U1 / 2

    def _onsite_upper_b(self, site) -> float:
        """Energia on-site dla atomów B w górnej warstwie."""
        x, y = site.pos
        # Górna warstwa: oba podwęzły mają tę samą energię warstwową
        return self.params.V1 - self.params.U1 / 2

    def _junction_VU(self, x: float) -> tuple:
        """
        Oblicz potencjał V i przerwę U w pozycji x — profil tanh(x/d).

        V(x) = (V1+V2)/2 + (V1-V2)/2 * tanh(x/d)
        U(x) = (U1+U2)/2 + (U1-U2)/2 * tanh(x/d)

        Granice: x → +∞ → (V1, U1),  x → -∞ → (V2, U2).
        Szerokość przejścia: d = params.d  [j.a.].
        """
        t = np.tanh(x / self.params.d)
        V = (self.params.V1 + self.params.V2) / 2 + (self.params.V1 - self.params.V2) / 2 * t
        U = (self.params.U1 + self.params.U2) / 2 + (self.params.U1 - self.params.U2) / 2 * t
        return V, U

    def _onsite_lower_a_profile(self, site) -> float:
        """Energia on-site z profilem złącza dla atomów A dolnych."""
        x, y = site.pos
        V, U = self._junction_VU(x)
        return V + U / 2

    def _onsite_lower_b_profile(self, site) -> float:
        """Energia on-site z profilem złącza dla atomów B dolnych."""
        x, y = site.pos
        V, U = self._junction_VU(x)
        return V + U / 2

    def _onsite_upper_a_profile(self, site) -> float:
        """Energia on-site z profilem złącza dla atomów A górnych."""
        x, y = site.pos
        V, U = self._junction_VU(x)
        return V - U / 2

    def _onsite_upper_b_profile(self, site) -> float:
        """Energia on-site z profilem złącza dla atomów B górnych."""
        x, y = site.pos
        V, U = self._junction_VU(x)
        return V - U / 2

    # --- Parametryczne on-site: V1,U1,V2,U2 przekazywane jako params Kwant ---

    def _onsite_lower_a_par(self, site, V1, U1, V2, U2) -> float:
        x, _ = site.pos
        t = np.tanh(x / self.params.d)
        V = (V1 + V2) / 2 + (V1 - V2) / 2 * t
        U = (U1 + U2) / 2 + (U1 - U2) / 2 * t
        return V + U / 2

    def _onsite_lower_b_par(self, site, V1, U1, V2, U2) -> float:
        return self._onsite_lower_a_par(site, V1, U1, V2, U2)

    def _onsite_upper_a_par(self, site, V1, U1, V2, U2) -> float:
        x, _ = site.pos
        t = np.tanh(x / self.params.d)
        V = (V1 + V2) / 2 + (V1 - V2) / 2 * t
        U = (U1 + U2) / 2 + (U1 - U2) / 2 * t
        return V - U / 2

    def _onsite_upper_b_par(self, site, V1, U1, V2, U2) -> float:
        return self._onsite_upper_a_par(site, V1, U1, V2, U2)

    # Lead on-site: lewy (x→−∞) używa V2,U2; prawy (x→+∞) używa V1,U1
    def _left_la_par(self, site, V2, U2) -> float: return V2 + U2 / 2
    def _left_lb_par(self, site, V2, U2) -> float: return V2 + U2 / 2
    def _left_ua_par(self, site, V2, U2) -> float: return V2 - U2 / 2
    def _left_ub_par(self, site, V2, U2) -> float: return V2 - U2 / 2
    def _right_la_par(self, site, V1, U1) -> float: return V1 + U1 / 2
    def _right_lb_par(self, site, V1, U1) -> float: return V1 + U1 / 2
    def _right_ua_par(self, site, V1, U1) -> float: return V1 - U1 / 2
    def _right_ub_par(self, site, V1, U1) -> float: return V1 - U1 / 2

    def _scat_hopping(self, site1, site2) -> complex:
        """Przeskok w obszarze centralnym — pole magnetyczne zmienia się płynnie."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos

        x_sr = 0.5 * (x1 + x2)
        y_mid = 0.5 * (y1 + y2)

        B = self.params.B * np.tanh(x_sr / self.params.d)

        integral = B * y_mid * (x2 - x1)

        return self.t_scaled * np.exp(1j * integral)

    def _scat_hopping_par(self, site1, site2, B) -> complex:
        """Parametryczny przeskok w obszarze centralnym (B przekazywane przez params)."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos

        x_sr = 0.5 * (x1 + x2)
        y_mid = 0.5 * (y1 + y2)

        B_loc = B * np.tanh(x_sr / self.params.d)
        integral = B_loc * y_mid * (x2 - x1)

        return self.t_scaled * np.exp(1j * integral)

    def _scat_hopping_uniform_B_par(self, site1, site2, B) -> complex:
        """Parametryczny hopping przy stałym polu B w całym układzie."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        y_mid = 0.5 * (y1 + y2)
        integral = B * y_mid * (x2 - x1)
        return self.t_scaled * np.exp(1j * integral)

    def _left_lead_hopping(self, site1, site2) -> complex:
        """Lewy przewodnik: pole B = 0. Brak fazy Peierlsa."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        x_sr = 0.5 * (x1 + x2)
        y_mid = 0.5 * (y1 + y2)
        B = -self.params.B

        integral = B * y_mid * (x2 - x1)

        return self.t_scaled * np.exp(1j * integral)

    def _left_lead_hopping_par(self, site1, site2, B) -> complex:
        """Parametryczny hopping lewego przewodnika (B przekazywane przez params)."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        x_sr = 0.5 * (x1 + x2)
        y_mid = 0.5 * (y1 + y2)
        B_loc = -B

        integral = B_loc * y_mid * (x2 - x1)

        return self.t_scaled * np.exp(1j * integral)

    def _right_lead_hopping(self, site1, site2) -> complex:
        """Prawy przewodnik: pole B = 0. Brak fazy Peierlsa."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        y_mid = 0.5 * (y1 + y2)
        B = self.params.B

        integral = B * y_mid * (x2 - x1)

        return self.t_scaled * np.exp(1j * integral)

    def _right_lead_hopping_par(self, site1, site2, B) -> complex:
        """Parametryczny hopping prawego przewodnika (B przekazywane przez params)."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        y_mid = 0.5 * (y1 + y2)
        B_loc = B

        integral = B_loc * y_mid * (x2 - x1)

        return self.t_scaled * np.exp(1j * integral)

    def _uniform_lead_hopping_par(self, site1, site2, B) -> complex:
        """Parametryczny hopping leadu przy tym samym stałym polu B."""
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        y_mid = 0.5 * (y1 + y2)
        integral = B * y_mid * (x2 - x1)
        return self.t_scaled * np.exp(1j * integral)

    def _interlayer_hopping(self, site1, site2) -> float:
        """Przeskok międzywarstwowy γ₁ (bez skalowania)."""
        return self.gamma1

    def build_system(self, use_potential_profile: bool = False) -> 'BilayerGrapheneSystem':
        """
        Zbuduj system kwantowy.
        
        Args:
            use_potential_profile: Czy użyć profilu potencjału tanh(x/d)
            
        Returns:
            self dla łańcuchowania metod
        """
        self._use_profile = use_potential_profile
        syst = kwant.Builder()

        # Wybór funkcji on-site
        if use_potential_profile:
            onsite_low_a = self._onsite_lower_a_profile
            onsite_low_b = self._onsite_lower_b_profile
            onsite_upp_a = self._onsite_upper_a_profile
            onsite_upp_b = self._onsite_upper_b_profile
        else:
            onsite_low_a = self._onsite_lower_a
            onsite_low_b = self._onsite_lower_b
            onsite_upp_a = self._onsite_upper_a
            onsite_upp_b = self._onsite_upper_b

        # Dolna warstwa
        syst[self.low_a.shape(self._rect_shape, (0, 0))] = onsite_low_a
        syst[self.low_b.shape(self._rect_shape, (0, 0))] = onsite_low_b

        # Górna warstwa
        syst[self.upp_a.shape(self._rect_shape, (0, 0))] = onsite_upp_a
        syst[self.upp_b.shape(self._rect_shape, (0, 0))] = onsite_upp_b

        # Przeskoki wewnątrzwarstwowe
        syst[self.lower_lattice.neighbors()] = self._scat_hopping
        syst[self.upper_lattice.neighbors()] = self._scat_hopping

        # Przeskok międzywarstwowy: B dolnej ↔ A górnej
        syst[kwant.builder.HoppingKind((0, 0), self.upp_a, self.low_b)] = self._interlayer_hopping

        self._builder = syst
        return self

    def attach_leads(self) -> 'BilayerGrapheneSystem':
        """
        Dodaj przewodniki (leads) do systemu.
        
        Używamy symetrii translacyjnej zgodnej z wektorami Bravais'a siatki.
        
        Returns:
            self dla łańcuchowania metod
        """
        a = self.a

        # Energie on-site w przewodnikach (płaski potencjał)
        # Prawa strona złącza (x→+∞): V1, U1
        V_right = self.params.V1
        U_right = self.params.U1
        V_left = self.params.V2
        U_left = self.params.U2

        # Wektor translacji dla leadu wzdłuż x:
        # (√3·a, 0) = 2·v2 − v1, gdzie v1=(0,a), v2=(√3/2·a, a/2) — poprawny wektor sieci
        lead_trans = np.array([np.sqrt(3.0) * self.a, 0.0])

        # === LEWY PRZEWODNIK ===
        sym_left = kwant.TranslationalSymmetry(-lead_trans)
        left_lead = kwant.Builder(sym_left)

        # Dolna warstwa
        left_lead[self.low_a.shape(self._lead_shape, (-self.params.L, 0))] = V_left + U_left / 2
        left_lead[self.low_b.shape(self._lead_shape, (-self.params.L, 0))] = V_left + U_left / 2

        # Górna warstwa
        left_lead[self.upp_a.shape(self._lead_shape, (-self.params.L, 0))] = V_left - U_left / 2
        left_lead[self.upp_b.shape(self._lead_shape, (-self.params.L, 0))] = V_left - U_left / 2

        # Przeskoki
        left_lead[self.lower_lattice.neighbors()] = self._left_lead_hopping
        left_lead[self.upper_lattice.neighbors()] = self._left_lead_hopping
        left_lead[kwant.builder.HoppingKind((0, 0), self.upp_a, self.low_b)] = self._interlayer_hopping

        # === PRAWY PRZEWODNIK ===
        sym_right = kwant.TranslationalSymmetry(lead_trans)
        right_lead = kwant.Builder(sym_right)

        # Dolna warstwa
        right_lead[self.low_a.shape(self._lead_shape, (self.params.L, 0))] = V_right + U_right / 2
        right_lead[self.low_b.shape(self._lead_shape, (self.params.L, 0))] = V_right + U_right / 2

        # Górna warstwa
        right_lead[self.upp_a.shape(self._lead_shape, (self.params.L, 0))] = V_right - U_right / 2
        right_lead[self.upp_b.shape(self._lead_shape, (self.params.L, 0))] = V_right - U_right / 2

        # Przeskoki
        right_lead[self.lower_lattice.neighbors()] = self._right_lead_hopping
        right_lead[self.upper_lattice.neighbors()] = self._right_lead_hopping
        right_lead[kwant.builder.HoppingKind((0, 0), self.upp_a, self.low_b)] = self._interlayer_hopping

        # Dodanie przewodników
        self._builder.attach_lead(left_lead)
        self._builder.attach_lead(right_lead)

        return self

    def build_parametric_system(self, uniform_B: bool = False) -> 'BilayerGrapheneSystem':
        """
        Buduje system z parametrycznymi energiami on-site (profil tanh)
        oraz parametrycznym polem magnetycznym B.

        V1, U1, V2, U2, B są przekazywane jako params do kwant.smatrix().
        Użycie: buduj system raz i reużywaj w pętlach po B i/lub Vb.
        """
        syst = kwant.Builder()
        syst[self.low_a.shape(self._rect_shape, (0, 0))] = self._onsite_lower_a_par
        syst[self.low_b.shape(self._rect_shape, (0, 0))] = self._onsite_lower_b_par
        syst[self.upp_a.shape(self._rect_shape, (0, 0))] = self._onsite_upper_a_par
        syst[self.upp_b.shape(self._rect_shape, (0, 0))] = self._onsite_upper_b_par
        hopping = (self._scat_hopping_uniform_B_par if uniform_B
               else self._scat_hopping_par)
        syst[self.lower_lattice.neighbors()] = hopping
        syst[self.upper_lattice.neighbors()] = hopping
        syst[kwant.builder.HoppingKind((0, 0), self.upp_a, self.low_b)] = self._interlayer_hopping
        self._builder = syst
        return self

    def attach_parametric_leads(self, uniform_B: bool = False) -> 'BilayerGrapheneSystem':
        """Dodaj przewodniki z parametrycznymi energiami on-site."""
        lead_trans = np.array([np.sqrt(3.0) * self.a, 0.0])

        sym_left = kwant.TranslationalSymmetry(-lead_trans)
        left_lead = kwant.Builder(sym_left)
        left_lead[self.low_a.shape(self._lead_shape, (-self.params.L, 0))] = self._left_la_par
        left_lead[self.low_b.shape(self._lead_shape, (-self.params.L, 0))] = self._left_lb_par
        left_lead[self.upp_a.shape(self._lead_shape, (-self.params.L, 0))] = self._left_ua_par
        left_lead[self.upp_b.shape(self._lead_shape, (-self.params.L, 0))] = self._left_ub_par
        lead_hopping = (self._uniform_lead_hopping_par if uniform_B
                else self._left_lead_hopping_par)
        left_lead[self.lower_lattice.neighbors()] = lead_hopping
        left_lead[self.upper_lattice.neighbors()] = lead_hopping
        left_lead[kwant.builder.HoppingKind((0, 0), self.upp_a, self.low_b)] = self._interlayer_hopping

        sym_right = kwant.TranslationalSymmetry(lead_trans)
        right_lead = kwant.Builder(sym_right)
        right_lead[self.low_a.shape(self._lead_shape, (self.params.L, 0))] = self._right_la_par
        right_lead[self.low_b.shape(self._lead_shape, (self.params.L, 0))] = self._right_lb_par
        right_lead[self.upp_a.shape(self._lead_shape, (self.params.L, 0))] = self._right_ua_par
        right_lead[self.upp_b.shape(self._lead_shape, (self.params.L, 0))] = self._right_ub_par
        right_lead[self.lower_lattice.neighbors()] = lead_hopping
        right_lead[self.upper_lattice.neighbors()] = lead_hopping
        right_lead[kwant.builder.HoppingKind((0, 0), self.upp_a, self.low_b)] = self._interlayer_hopping

        self._builder.attach_lead(left_lead)
        self._builder.attach_lead(right_lead)
        return self

    def finalize(self) -> 'BilayerGrapheneSystem':
        """
        Finalizuj system kwantowy.
        
        Returns:
            self dla łańcuchowania metod
        """
        self._system = self._builder.finalized()
        return self

    @property
    def system(self) -> kwant.builder.FiniteSystem:
        """Zwróć sfinalizowany system Kwant."""
        if self._system is None:
            raise RuntimeError("System nie został sfinalizowany. Wywołaj finalize() najpierw.")
        return self._system

    @property
    def builder(self) -> kwant.Builder:
        """Zwróć builder systemu."""
        return self._builder


def make_blg_system(params: BLGSystemParameters, 
                    use_potential_profile: bool = False) -> BilayerGrapheneSystem:
    """
    Funkcja pomocnicza do szybkiego tworzenia systemu BLG.
    
    Args:
        params: Parametry systemu
        use_potential_profile: Czy użyć profilu potencjału tanh
        
    Returns:
        Sfinalizowany system BLG
    """
    blg = BilayerGrapheneSystem(params)
    blg.build_system(use_potential_profile)
    blg.attach_leads()
    blg.finalize()
    return blg


def make_blg_parametric_system(params: BLGSystemParameters,
                               uniform_B: bool = False) -> BilayerGrapheneSystem:
    """
    Tworzy system BLG z parametrycznymi energiami on-site (V1, U1, V2, U2)
    oraz parametrycznym polem B.

    System można zbudować raz i reużywać w pętlach po B i Vb::

        blg = make_blg_parametric_system(p_b)   # raz
        fsyst = blg.system
        for B, V1, U1, V2, U2 in ...:
            sm = kwant.smatrix(fsyst, energy,
                               params=dict(B=B, V1=V1, U1=U1, V2=V2, U2=U2))
            G = sm.transmission(1, 0)
    """
    blg = BilayerGrapheneSystem(params)
    blg.build_parametric_system(uniform_B=uniform_B)
    blg.attach_parametric_leads(uniform_B=uniform_B)
    blg.finalize()
    return blg


def make_blg_unit_cell(params: BLGSystemParameters) -> kwant.Builder:
    """
    Utwórz komórkę elementarną BLG do obliczeń struktury pasmowej.
    
    Args:
        params: Parametry systemu
        
    Returns:
        Builder z komórką elementarną (z symetrią translacyjną)
    """
    blg_temp = BilayerGrapheneSystem(params)
    a = blg_temp.a
    a_cc = a / np.sqrt(3.0)

    # Wektory translacji - zgodne z _create_lattices (armchair wzdłuż x)
    bravais_vectors = [
        (0, a),
        (np.sqrt(3.0) / 2 * a, 0.5 * a),
    ]

    syst = kwant.Builder(kwant.TranslationalSymmetry(*bravais_vectors))
    
    # Energie on-site
    syst[blg_temp.low_a(0, 0)] = params.V1 + params.U1 / 2
    syst[blg_temp.low_b(0, 0)] = params.V1 - params.U1 / 2
    syst[blg_temp.upp_a(0, 0)] = params.V1 + params.U1 / 2
    syst[blg_temp.upp_b(0, 0)] = params.V1 - params.U1 / 2
    
    # Przeskoki wewnątrzwarstwowe (skalowane)
    t_scaled = params.t / params.s_f
    syst[blg_temp.lower_lattice.neighbors()] = t_scaled
    syst[blg_temp.upper_lattice.neighbors()] = t_scaled
    
    # Przeskok międzywarstwowy (bez skalowania)
    syst[kwant.builder.HoppingKind((0, 0), blg_temp.upp_a, blg_temp.low_b)] = params.gamma1
    
    return syst


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from kwant.physics import Bands
    
    print("Tworzenie systemu dwuwarstwowego grafenu...")
    
    # Parametry testowe
    params = BLGSystemParameters(
        L=30.0 / l_scale,
        W=25.0 / l_scale,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        s_f=8.0,
        V1=0.0,
        U1=0.05 / E_scale,  # mała przerwa energetyczna
        B=0.0,
        name="test_BLG"
    )
    
    # Utwórz system
    blg = make_blg_system(params)
    print(f"System utworzony: {blg.system.graph.num_nodes} węzłów")
    
    # Oblicz strukturę pasmową
    print("\nObliczanie struktury pasmowej...")
    bands = Bands(blg.system.leads[1])
    k_arr = np.linspace(-0.5, 0.5, 200)
    momenta = k_arr * (params.s_f * A_GRAPHENE / l_scale)
    energies = [bands(k) for k in momenta]
    energies = np.array(energies) * E_scale
    
    # Wykres
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Struktura pasmowa
    ax1.plot(k_arr, energies)
    ax1.set_xlabel("k (1/a)")
    ax1.set_ylabel("E (eV)")
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_title(f"Struktura pasmowa BLG\nU = {params.U1 * E_scale:.3f} eV")
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Wizualizacja systemu
    kwant.plot(blg.system, ax=ax2, show=False)
    ax2.set_title("System BLG")
    
    plt.tight_layout()
    plt.savefig("plots/BLG_test.png", dpi=150)
    plt.show()
    
    print("\nTest zakończony pomyślnie!")
