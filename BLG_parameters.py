"""
Moduł do rozwiązywania układu równań samouzgodnionych dla podwójnego
dwuwarstwowego grafenu (double bilayer graphene).

Równania oparte na dokumencie teoretycznym - układ 8 równań nieliniowych
dla zmiennych: n1, dn1, U1, Vg1, n2, dn2, U2, Vg2

Parametry pojemności:
- Cm: pojemność między bilayerami (przez hBN ~80nm)
- Cb: pojemność bramki dolnej (SiO2 ~330nm)
- Ct: pojemność bramki górnej (jeśli obecna)
- Cg: pojemność między warstwami w bilayerze (d_G ~ 0.12nm)
"""

import numpy as np
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class CapacitanceParameters:
    """
    Parametry pojemnościowe dla podwójnego dwuwarstwowego grafenu.
    
    Wszystkie pojemności w jednostkach [10^15 m^-2 V^-1] (podzielone przez e)
    
    Attributes:
        Cm: Pojemność między bilayerami (hBN, ~80nm) ≈ 2.55
        Cb: Pojemność bramki dolnej (SiO2, ~330nm) ≈ 0.653
        Ct: Pojemność bramki górnej (opcjonalna) ≈ 0 lub wartość
        Cg: Pojemność między warstwami w bilayerze ≈ 230 lub 460.5
        
    Wzory na pojemności:
        C/e = ε₀ * ε_r / (e * d)
        gdzie ε₀ = 8.854e-12 F/m, e = 1.602e-19 C
    """
    # Pojemność między bilayerami (hBN, ε=3.7, d=80nm)
    Cm: float = 2.55  # [10^15 m^-2 V^-1]
    
    # Pojemność bramki dolnej (SiO2, ε=3.9, d=330nm) 
    Cb: float = 0.653  # [10^15 m^-2 V^-1]
    
    # Pojemność bramki górnej (domyślnie 0 = brak bramki górnej)
    Ct: float = 0.0  # [10^15 m^-2 V^-1]
    
    # Pojemność między warstwami w bilayerze (próżnia, ε=1, d=0.12nm)
    # Uwaga: w eksperymencie może być ~230, w obliczeniach czasem 460.5
    Cg: float = 230.0  # [10^15 m^-2 V^-1]
    
    @classmethod
    def calculate_capacitance(cls, epsilon_r: float, d_nm: float) -> float:
        """
        Oblicz pojemność na jednostkę ładunku.
        
        Args:
            epsilon_r: Względna przenikalność elektryczna
            d_nm: Grubość dielektryka w nanometrach
            
        Returns:
            C/e w jednostkach [10^15 m^-2 V^-1]
        """
        epsilon_0 = 8.854e-12  # F/m
        e = 1.602e-19  # C
        d = d_nm * 1e-9  # m
        return epsilon_0 * epsilon_r / (e * d) * 1e-15


@dataclass
class BLGPhysicsParameters:
    """
    Parametry fizyczne dla dwuwarstwowego grafenu.
    
    Attributes:
        gamma1: Energia przeskoku między warstwami [eV]
        hvf2: (ħ*v_F)^2 gdzie v_F to prędkość Fermiego [eV^2 * nm^2]
        n_t: Charakterystyczna gęstość nośników [10^15 m^-2]
    """
    # Energia przeskoku między warstwami (gamma_1)
    gamma1: float = 0.39  # eV
    
    # (ħ*v_F)^2 - związane z prędkością Fermiego
    # v_F ≈ 10^6 m/s dla grafenu, ħv_F ≈ 0.658 eV*nm
    hvf2: float = 6.39**2  # (eV*nm)^2 ≈ 40.83
    
    # Charakterystyczna gęstość nośników
    n_t: float = 118.57  # [10^15 m^-2]


@dataclass 
class DoubleBLGState:
    """
    Stan podwójnego dwuwarstwowego grafenu - wynik rozwiązania równań.
    
    Attributes:
        n1, n2: Całkowite gęstości nośników w bilayer 1 i 2
        dn1, dn2: Różnice gęstości między warstwami (asymetria)
        U1, U2: Przerwy energetyczne (asymmetry gap) [eV]
        Vg1, Vg2: Potencjały bramkowe bilayerów [V]
    """
    n1: float = 0.0   # gęstość w bilayer 1 [10^15 m^-2]
    dn1: float = 0.0  # asymetria gęstości w bilayer 1
    U1: float = 0.0   # przerwa energetyczna bilayer 1 [eV]
    Vg1: float = 0.0  # potencjał bramkowy bilayer 1 [V]
    
    n2: float = 0.0   # gęstość w bilayer 2 [10^15 m^-2]
    dn2: float = 0.0  # asymetria gęstości w bilayer 2
    U2: float = 0.0   # przerwa energetyczna bilayer 2 [eV]
    Vg2: float = 0.0  # potencjał bramkowy bilayer 2 [V]
    
    def as_array(self) -> np.ndarray:
        """Zwróć stan jako tablicę numpy."""
        return np.array([self.n1, self.dn1, self.U1, self.Vg1,
                        self.n2, self.dn2, self.U2, self.Vg2])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'DoubleBLGState':
        """Utwórz stan z tablicy numpy."""
        return cls(n1=arr[0], dn1=arr[1], U1=arr[2], Vg1=arr[3],
                  n2=arr[4], dn2=arr[5], U2=arr[6], Vg2=arr[7])


class DoubleBLGSolver:
    """
    Solver dla układu równań samouzgodnionych podwójnego dwuwarstwowego grafenu.
    
    Rozwiązuje układ 8 równań nieliniowych dla zadanych napięć bramek Vt i Vb.
    
    Równania (bazowane na dokumencie teoretycznym):
    - f1t, f3t: Równania na gęstości dla górnego bilayera
    - f2t: Równanie na asymetrię dla górnego bilayera  
    - f4t: Równanie na potencjał chemiczny górnego bilayera
    - f1b, f3b: Równania na gęstości dla dolnego bilayera
    - f2b: Równanie na asymetrię dla dolnego bilayera
    - f4b: Równanie na potencjał chemiczny dolnego bilayera
    """
    
    def __init__(self, 
                 cap_params: Optional[CapacitanceParameters] = None,
                 phys_params: Optional[BLGPhysicsParameters] = None):
        """
        Inicjalizacja solvera.
        
        Args:
            cap_params: Parametry pojemnościowe (domyślnie standardowe)
            phys_params: Parametry fizyczne BLG (domyślnie standardowe)
        """
        self.cap = cap_params or CapacitanceParameters()
        self.phys = phys_params or BLGPhysicsParameters()
        
        # Wbudowane przesunięcia gęstości (intrinsic doping)
        self.n0t: float = 0.0  # przesunięcie dla górnego bilayera
        self.n0b: float = 0.0  # przesunięcie dla dolnego bilayera
        self.dn0t: float = 0.0  # przesunięcie asymetrii górnego
        self.dn0b: float = 0.0  # przesunięcie asymetrii dolnego
        
    def _f1t(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie na asymetrię gęstości górnego bilayera (równ. 41)."""
        return (-dn1 + self.dn0t 
                + self.cap.Cm * ((Vg2 - U2/2) - (Vg1 + U1/2)) 
                - self.cap.Ct * (Vt - (Vg1 - U1/2)) 
                - 2 * self.cap.Cg * U1)
    
    def _f2t(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie samouzgodnione na asymetrię górnego bilayera."""
        gamma1 = self.phys.gamma1
        n_t = self.phys.n_t
        
        # Zabezpieczenie przed log(0)
        n1_safe = max(abs(n1), 1e-10)
        
        arg_log = n1_safe / n_t / 2 + 0.5 * np.sqrt((n1 / n_t)**2 + (U1 / 2 / gamma1)**2)
        arg_log = max(arg_log, 1e-10)
        
        return dn1 + n_t / 2 / gamma1 * U1 * np.log(arg_log)
    
    def _f3t(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie na całkowitą gęstość górnego bilayera."""
        return (-n1 + self.n0t 
                + self.cap.Cm * ((Vg2 - U2/2) - (Vg1 + U1/2)) 
                + self.cap.Ct * (Vt - (Vg1 - U1/2)))
    
    def _f4t(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie na potencjał chemiczny górnego bilayera."""
        gamma1 = self.phys.gamma1
        hvf2 = self.phys.hvf2
        
        n1_abs = abs(n1) + 1e-10
        
        # Energia poziomy Fermiego dla BLG
        inner_sqrt = gamma1**2 + (4 * hvf2 * 1e-5 * np.pi) * n1_abs * (1 + (U1/gamma1)**2)
        inner_sqrt = max(inner_sqrt, 0)
        
        outer = (gamma1**2/2 + U1**2/4 
                + (hvf2 * 1e-5 * np.pi) * n1_abs 
                - gamma1/2 * np.sqrt(inner_sqrt))
        outer = max(outer, 0)
        
        sign_n1 = -1 if n1 >= 0 else 1
        return Vg1 + np.sqrt(outer) * sign_n1
    
    def _f1b(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie na asymetrię gęstości dolnego bilayera."""
        return (-dn2 + self.dn0b 
                + self.cap.Cb * (Vb - (Vg2 + U2/2)) 
                - self.cap.Cm * ((Vg1 + U1/2) - (Vg2 - U2/2)) 
                - 2 * self.cap.Cg * U2)
    
    def _f2b(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie samouzgodnione na asymetrię dolnego bilayera."""
        gamma1 = self.phys.gamma1
        n_t = self.phys.n_t
        
        n2_safe = max(abs(n2), 1e-10)
        
        arg_log = n2_safe / n_t / 2 + 0.5 * np.sqrt((n2 / n_t)**2 + (U2 / 2 / gamma1)**2)
        arg_log = max(arg_log, 1e-10)
        
        return dn2 + n_t / 2 / gamma1 * U2 * np.log(arg_log)
    
    def _f3b(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie na całkowitą gęstość dolnego bilayera."""
        return (-n2 
                + self.cap.Cb * (Vb - (Vg2 + U2/2)) 
                + self.cap.Cm * ((Vg1 + U1/2) - (Vg2 - U2/2)))
    
    def _f4b(self, n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb) -> float:
        """Równanie na potencjał chemiczny dolnego bilayera."""
        gamma1 = self.phys.gamma1
        hvf2 = self.phys.hvf2
        
        n2_abs = abs(n2) + 1e-10
        
        inner_sqrt = gamma1**2 + (4 * hvf2 * 1e-5 * np.pi) * n2_abs * (1 + (U2/gamma1)**2)
        inner_sqrt = max(inner_sqrt, 0)
        
        outer = (gamma1**2/2 + U2**2/4 
                + (hvf2 * 1e-5 * np.pi) * n2_abs 
                - gamma1/2 * np.sqrt(inner_sqrt))
        outer = max(outer, 0)
        
        sign_n2 = -1 if n2 >= 0 else 1
        return Vg2 + np.sqrt(outer) * sign_n2
    
    def equations(self, x: np.ndarray, Vt: float, Vb: float) -> np.ndarray:
        """
        Układ 8 równań do rozwiązania.
        
        Args:
            x: Wektor zmiennych [n1, dn1, U1, Vg1, n2, dn2, U2, Vg2]
            Vt: Napięcie bramki górnej [V]
            Vb: Napięcie bramki dolnej [V]
            
        Returns:
            Wektor residuów równań
        """
        n1, dn1, U1, Vg1, n2, dn2, U2, Vg2 = x
        
        return np.array([
            self._f1t(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f2t(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f3t(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f4t(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f1b(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f2b(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f3b(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
            self._f4b(n1, dn1, U1, Vg1, n2, dn2, U2, Vg2, Vt, Vb),
        ])
    
    def solve(self, Vt: float, Vb: float, 
              initial_guess: Optional[np.ndarray] = None) -> DoubleBLGState:
        """
        Rozwiąż układ równań dla zadanych napięć bramek.
        
        Args:
            Vt: Napięcie bramki górnej [V]
            Vb: Napięcie bramki dolnej [V]
            initial_guess: Początkowe wartości zmiennych (opcjonalne)
            
        Returns:
            DoubleBLGState ze znalezionym rozwiązaniem
        """
        if initial_guess is None:
            # Domyślne wartości początkowe - skalowane z Vb
            scale = max(abs(Vb), 1.0)
            initial_guess = np.array([
                0.1 * scale,    # n1 - skalowane z napięciem
                0.01 * scale,   # dn1
                0.01,           # U1
                0.1 * Vb if abs(Vb) > 0.1 else 0.01,  # Vg1
                0.1 * scale,    # n2 - skalowane z napięciem
                0.01 * scale,   # dn2
                0.01,           # U2
                0.1 * Vb if abs(Vb) > 0.1 else 0.01,  # Vg2
            ])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution, info, ier, mesg = fsolve(
                self.equations, 
                initial_guess, 
                args=(Vt, Vb),
                full_output=True,
                xtol=1e-10,
                maxfev=5000
            )
        
        # Sprawdź zbieżność przez residuum
        residual = np.max(np.abs(self.equations(solution, Vt, Vb)))
        if ier != 1 and residual > 1e-6:
            warnings.warn(f"Solver nie zbiegł dla Vt={Vt}, Vb={Vb}: residuum={residual:.2e}")
        
        return DoubleBLGState.from_array(solution)
    
    def solve_sweep(self, Vt_arr: np.ndarray, Vb_arr: np.ndarray,
                    use_previous: bool = True) -> dict:
        """
        Rozwiąż dla zakresu napięć bramek.
        
        Args:
            Vt_arr: Tablica napięć bramki górnej [V]
            Vb_arr: Tablica napięć bramki dolnej [V]
            use_previous: Użyj poprzedniego rozwiązania jako punktu startowego
            
        Returns:
            Słownik z tablicami wyników
        """
        results = {
            'Vt': Vt_arr,
            'Vb': Vb_arr,
            'n1': np.zeros((len(Vt_arr), len(Vb_arr))),
            'dn1': np.zeros((len(Vt_arr), len(Vb_arr))),
            'U1': np.zeros((len(Vt_arr), len(Vb_arr))),
            'Vg1': np.zeros((len(Vt_arr), len(Vb_arr))),
            'n2': np.zeros((len(Vt_arr), len(Vb_arr))),
            'dn2': np.zeros((len(Vt_arr), len(Vb_arr))),
            'U2': np.zeros((len(Vt_arr), len(Vb_arr))),
            'Vg2': np.zeros((len(Vt_arr), len(Vb_arr))),
        }
        
        for i, Vt in enumerate(Vt_arr):
            # Reset punktu startowego dla każdego nowego wiersza Vt,
            # żeby nie przenosić złych wartości z końca poprzedniego wiersza.
            prev_solution = None
            for j, Vb in enumerate(Vb_arr):
                use_prev = (use_previous
                            and prev_solution is not None
                            and abs(prev_solution[4]) > 0.05 * max(abs(Vb), 1.0) * self.cap.Cb)
                if use_prev:
                    state = self.solve(Vt, Vb, prev_solution)
                else:
                    state = self.solve(Vt, Vb)
                
                prev_solution = state.as_array()
                
                results['n1'][i, j] = state.n1
                results['dn1'][i, j] = state.dn1
                results['U1'][i, j] = state.U1
                results['Vg1'][i, j] = state.Vg1
                results['n2'][i, j] = state.n2
                results['dn2'][i, j] = state.dn2
                results['U2'][i, j] = state.U2
                results['Vg2'][i, j] = state.Vg2
        
        return results
    
    def solve_1D_sweep(self, Vb_arr: np.ndarray, Vt: float = 0.0,
                       use_previous: bool = True) -> dict:
        """
        Rozwiąż dla zakresu napięć bramki dolnej przy stałym Vt.
        
        Args:
            Vb_arr: Tablica napięć bramki dolnej [V]
            Vt: Stałe napięcie bramki górnej [V]
            use_previous: Użyj poprzedniego rozwiązania jako punktu startowego
            
        Returns:
            Słownik z tablicami wyników
        """
        results = {
            'Vt': Vt,
            'Vb': Vb_arr.copy(),
            'n1': np.zeros(len(Vb_arr)),
            'dn1': np.zeros(len(Vb_arr)),
            'U1': np.zeros(len(Vb_arr)),
            'Vg1': np.zeros(len(Vb_arr)),
            'n2': np.zeros(len(Vb_arr)),
            'dn2': np.zeros(len(Vb_arr)),
            'U2': np.zeros(len(Vb_arr)),
            'Vg2': np.zeros(len(Vb_arr)),
        }
        
        prev_solution = None
        
        for i, Vb in enumerate(Vb_arr):
            # Przy przejściu przez Vb=0 poprzednie rozwiązanie ma n2≈0,
            # co powoduje rozbieżność Jakobianu (energia Fermiego ~√n).
            # Resetujemy punkt startowy gdy n2 poprzedniego kroku jest blisko 0.
            use_prev = (use_previous
                        and prev_solution is not None
                        and abs(prev_solution[4]) > 0.05 * max(abs(Vb), 1.0) * self.cap.Cb)
            if use_prev:
                state = self.solve(Vt, Vb, prev_solution)
            else:
                state = self.solve(Vt, Vb)
            
            prev_solution = state.as_array()
            
            results['n1'][i] = state.n1
            results['dn1'][i] = state.dn1
            results['U1'][i] = state.U1
            results['Vg1'][i] = state.Vg1
            results['n2'][i] = state.n2
            results['dn2'][i] = state.dn2
            results['U2'][i] = state.U2
            results['Vg2'][i] = state.Vg2
        
        return results


def example_usage():
    """Przykład użycia solvera."""
    # Utwórz solver ze standardowymi parametrami
    solver = DoubleBLGSolver()
    
    # Pojedyncze rozwiązanie
    state = solver.solve(Vt=0.0, Vb=10.0)
    print(f"Dla Vt=0, Vb=10V:")
    print(f"  n1={state.n1:.4f}, U1={state.U1:.4f} eV, Vg1={state.Vg1:.4f} V")
    print(f"  n2={state.n2:.4f}, U2={state.U2:.4f} eV, Vg2={state.Vg2:.4f} V")
    
    # Sweep po Vb
    Vb_arr = np.linspace(-60, 60, 61)
    results = solver.solve_1D_sweep(Vb_arr, Vt=0.0)
    
    print(f"\nSweep Vb od -60V do 60V:")
    print(f"  U1 range: [{results['U1'].min():.4f}, {results['U1'].max():.4f}] eV")
    print(f"  U2 range: [{results['U2'].min():.4f}, {results['U2'].max():.4f}] eV")
    
    return results


if __name__ == "__main__":
    results = example_usage()
