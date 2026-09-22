# Magisterka — Symulacje transportu w dwuwarstwowym grafenie (BLG)

Projekt zawiera zestaw skryptów Python do symulacji transportu kwantowego w
dwuwarstwowym grafenie (bilayer graphene, BLG) w konfiguracji Bernala (AB),
z wykorzystaniem pakietu [Kwant](https://kwant-project.org/). Badane
zagadnienia obejmują m.in.:

- strukturę pasmową (dyspersję) BLG z przerwą energetyczną indukowaną polem
  bramkującym,
- przewodność dwukońcówkową $G$ w funkcji pola magnetycznego $B$ i napięć
  bramek $V_t$, $V_b$ (mapy 2D),
- samouzgodniony model elektrostatyczny (Hartree/pojemnościowy) gęstości
  nośników i przerwy energetycznej,
- lokalny prąd wiązaniowy (bond current) i jego wizualizację,
- półklasyczne trajektorie stanów "snake" (cyklotronowe łuki) na złączu
  n-p/p-n.

Obliczenia są zaprojektowane pod uruchamianie na klastrze HPC (PLGrid, SLURM)
i podzielone na etapy: obliczenia (`main_BLG_data.py`) → wykresy
(`main_BLG_plots.py`) → analiza (`analyze_task8_*.py`).

## Spis treści

- [Struktura projektu](#struktura-projektu)
- [Moduły bazowe](#moduły-bazowe)
- [Skrypty główne (zadania)](#skrypty-główne-zadania)
- [Skrypty pomocnicze / analiza](#skrypty-pomocnicze--analiza)
- [Katalogi z danymi i wykresami](#katalogi-z-danymi-i-wykresami)
- [Uruchamianie na klastrze SLURM](#uruchamianie-na-klastrze-slurm)
- [Wymagania i instalacja](#wymagania-i-instalacja)
- [Konwencje w kodzie](#konwencje-w-kodzie)

## Struktura projektu

```
BLG_parameters.py           # samouzgodniony solver elektrostatyczny (double-bilayer)
BLG_system.py                # budowa sieci Kwant (geometria, leady, hamiltonian)
BLG_transport.py             # dyspersja, transmisja, przewodność, prąd lokalny
Constants.py                 # stałe fizyczne i przeliczniki jednostek
parameters_BLG_plus_offset.py# skrypt roboczy/testowy dla solvera z offsetem

main_BLG.py                  # przykładowe/demo zadania (dyspersja, G(B), prąd)
main_BLG_data.py             # etap 1: obliczenia wszystkich "zadań" (cache do data/)
main_BLG_plots.py            # etap 2: wykresy na podstawie danych z data/
main_BLG_task_F.py           # zadanie F: dyspersja w pojedynczym punkcie (B, Vt, Vb)
main_BLG_task8_chunk.py      # zadanie 8 w kawałkach (chunk) — mapa G(B, Vb)
main_BLG_task9.py            # zadanie 9: lokalny prąd w pojedynczym punkcie
main_BLG_trajectories.py     # trajektorie stanów snake (półklasyczne)
merge_BLG_task8_chunks.py    # scalanie chunków zadania 8 w jedną mapę

analyze_task8_line_extrema.py        # ekstrema na przekrojach G(B) przy stałym Vb
analyze_task8_map_fixed_B_extrema.py # ekstrema na przekrojach G(Vb) przy stałym B
replot_trajectories_AB.py            # ponowne rysowanie trajektorii z zapisanych .npz

run_*.sh, submit_*.sh        # skrypty SLURM do uruchamiania powyższych na klastrze

data/, data_d=*/             # cache wyników numerycznych (.npy/.npz)
plots/, plots_d=*/           # wygenerowane wykresy (.png/.pdf)
```

Warianty katalogów `data_d=0_AB`, `data_d=60`, `data_d=60_AB` (oraz
analogiczne `plots_d=*`) odpowiadają różnym szerokościom złącza `d`
(profil tanh między dwoma stanami gazu nośników, w nm) i różnym wariantom
geometrii/konfiguracji (`AB` — konfiguracja Bernala).

## Moduły bazowe

### [Constants.py](Constants.py)
Stałe fizyczne i przeliczniki jednostek atomowych (Hartree) używanych
wewnętrznie w obliczeniach ($\hbar = e = 1$):

- przeliczniki: `l_scale` (nm), `E_scale` (eV), `t_scale` (fs), `T_scale` (T),
- parametry sieci grafenu: `A_GRAPHENE_NM`, `A_CC_NM`,
- energie przeskoku: `T_HOPPING_EV`, `GAMMA1_EV`, `GAMMA3_EV`, `GAMMA4_EV`,
- parametry pojemnościowe: `C_INTERBILAYER`, `C_BACKGATE`, `C_INTRALAYER`,
  `N_CHARACTERISTIC`,
- klasa `SystemParameters` — starszy, uproszczony kontener parametrów układu.

### [BLG_parameters.py](BLG_parameters.py)
Samouzgodniony solver elektrostatyczny dla podwójnego dwuwarstwowego
grafenu (double bilayer graphene), sprzężonego pojemnościowo przez hBN i
bramki:

- `CapacitanceParameters` — pojemności `Cm` (między bilayerami), `Cb`
  (bramka dolna), `Ct` (bramka górna), `Cg` (między warstwami w bilayerze),
- `BLGPhysicsParameters` — `gamma1`, `hvf2` = $(\hbar v_F)^2$, `n_t`
  (charakterystyczna gęstość nośników),
- `DoubleBLGState` — stan wynikowy: `n1, dn1, U1, Vg1, n2, dn2, U2, Vg2`,
- `DoubleBLGSolver` — rozwiązuje układ 8 równań nieliniowych (`equations`,
  `solve`, `solve_1D_sweep`) metodą `scipy.optimize.fsolve`, zwracając
  gęstości, asymetrie i przerwy energetyczne w funkcji napięć bramek
  $V_t$, $V_b$.

### [BLG_system.py](BLG_system.py)
Budowa skończonego układu Kwant dla dwuwarstwowego grafenu w konfiguracji
Bernala AB:

- `BLGSystemParameters` — parametry geometrii i fizyki układu: `L`, `W`
  (rozmiary), `t`, `gamma1` (przeskoki), `s_f` (współczynnik skalowania
  sieci — "scalable tight-binding"), `B` (pole magnetyczne), `V1/U1/V2/U2`
  (potencjały/przerwy na złączu), `d` (szerokość złącza, profil tanh),
- `BilayerGrapheneSystem` — buduje sieci (`low_a`, `low_b`, `upp_a`,
  `upp_b`), energie własne (`_onsite_*`), profil złącza (`_junction_VU`)
  oraz finalizuje układ z leadami (`build()`),
- funkcje pomocnicze: `make_blg_system`, `make_blg_unit_cell` (komórka
  elementarna do struktury pasmowej), `make_blg_parametric_system`.

### [BLG_transport.py](BLG_transport.py)
Obliczenia transportowe na bazie układów z `BLG_system.py`:

- `calculate_dispersion`, `calculate_band_structure` — struktura pasmowa,
- `calculate_transmission`, `calculate_conductance` — macierz rozpraszania
  S, przewodność $G$ w jednostkach $2e^2/h$,
- `transmission_vs_energy`, `conductance_vs_magnetic_field` — krzywe
  $T(E)$, $G(B)$,
- `conductance_with_selfconsistent_params`, `conductance_map_G_B_Vb` —
  przewodność z parametrami z samouzgodnionego solvera, w tym pełna mapa
  2D $G(B, V_b)$,
- `calculate_total_bond_current` — lokalny prąd na wiązaniach sieci (do
  wizualizacji stanów snake),
- `ensure_data_dir`, `ensure_plots_dir` — pomocnicze tworzenie katalogów
  wyjściowych.

### [parameters_BLG_plus_offset.py](parameters_BLG_plus_offset.py)
Skrypt roboczy/eksploracyjny do testowania solvera samouzgodnionego z
dodatkowym przesunięciem (offsetem) — nieużywany przez pozostałe moduły.

## Skrypty główne (zadania)

Numeracja "zadanie N" odzwierciedla kolejne etapy pracy magisterskiej.

| Skrypt | Zadanie / cel | Wejście | Wyjście |
|---|---|---|---|
| [main_BLG.py](main_BLG.py) | Demonstracyjne odtworzenie zadań 1–9 w jednym przebiegu (dyspersja, G(B), prąd) | — | `data/`, `plots/` |
| [main_BLG_data.py](main_BLG_data.py) | **Etap 1**: liczy i cache'uje wyniki wszystkich zadań (`zadanie1`…`zadanie9`) bez rysowania | — | `data_d=*/*.npy`, `*.npz` |
| [main_BLG_plots.py](main_BLG_plots.py) | **Etap 2**: wczytuje dane z `data_d=*` i generuje finalne wykresy | `data_d=*/` | `plots_d=*/*.png`, `*.pdf` |
| [main_BLG_task_F.py](main_BLG_task_F.py) | Dyspersja w pojedynczym punkcie $(B, V_t, V_b)$ — do zrównoleglenia na klastrze | env: `B_T`, `VT_VALUE`, `VB_VALUE` | plik wyniku w `data_d=*/` |
| [main_BLG_task8_chunk.py](main_BLG_task8_chunk.py) | Fragment mapy $G(B, V_b)$ (zadanie 8) liczony w kawałkach po $B$ | env: zakresy `B_START/STOP/STEP`, `VB_START/STOP/STEP`, `VT_VALUE`, indeks chunku | `data_d=*/task8_chunks*/task8_chunk_XXXX_START_END.npz` |
| [merge_BLG_task8_chunks.py](merge_BLG_task8_chunks.py) | Scala chunki zadania 8 w jedną mapę 2D | `data_d=*/task8_chunks*/` | `zadanie8_G_map_*_full.npy` + metadane `.npz` |
| [main_BLG_task9.py](main_BLG_task9.py) | Lokalny prąd wiązaniowy w pojedynczym punkcie $(B, V_t, V_b)$ | env: `B_T`, `VT_VALUE`, `VB_VALUE` | `data_d=*/zadanie9_*/...` |
| [main_BLG_trajectories.py](main_BLG_trajectories.py) | Półklasyczne trajektorie stanów snake na profilu $n(x)$, $B(x)$ | CLI: `--vt`, `--d-nm`, `--x-limit-nm`, `--y-limit-nm`, `--out-dir` | `.npz` z trajektoriami w `--out-dir` |

Funkcje w `main_BLG_data.py` / `main_BLG_plots.py` obejmują m.in.:

- `zadanie1_dyspersja` — pasma $E(k)$ dla współczynników skalowania `s_f=1,4`,
- `zadanie2_dyspersja_z_przerwa` — pasma z przerwą $U$ i polem $B$,
- `zadanie3_przewodnosc_vs_B` — pojedyncza krzywa $G(B)$,
- `zadanie4_parametry_samouzgodnione` — sweep solvera (`n`, `U`, `Vg` vs $V_b$),
- `zadanie5_przewodnosc_samouzgodniona` — $G(V_b)$ z parametrami samouzgodnionymi,
- `zadanie6_mapy_2D` — mapa 2D solvera $(V_t, V_b) \to (n, U, V_g, \dots)$,
- `zadanie8_mapa_G_Vb_B_Vt0` — pełna mapa 2D $G(V_b, B)$ (najcięższe obliczenie, patrz chunking),
- `wykres_G_od_B_stale_Vb` — przekrój $G(B)$ przy ustalonym $V_b$,
- `zadanie9_prad_lokalny` — lokalny prąd $J_{ij}(E)$ przez macierz S.

## Skrypty pomocnicze / analiza

- [analyze_task8_line_extrema.py](analyze_task8_line_extrema.py) — wczytuje
  przekroje $G(B)$ przy stałym $V_b$, odszumia sygnał (filtr dolnoprzepustowy
  FFT) i znajduje ekstrema (`scipy.signal.find_peaks`), zapisuje wykresy z
  zaznaczonymi ekstremami.
- [analyze_task8_map_fixed_B_extrema.py](analyze_task8_map_fixed_B_extrema.py)
  — analogiczna analiza, ale na przekrojach $G(V_b)$ wyciętych z pełnej mapy
  2D przy zadanych wartościach $B$ (`--B 1.0 2.0 5.0 ...`).
- [replot_trajectories_AB.py](replot_trajectories_AB.py) — ponownie rysuje
  zapisane trajektorie stanów snake z ujednoliconym stylem/legendą.

## Katalogi z danymi i wykresami

- `data/`, `plots/` — domyślne katalogi wyjściowe (tworzone automatycznie).
- `data_d=0_AB/`, `data_d=60/`, `data_d=60_AB/` oraz odpowiadające im
  `plots_d=0_AB/`, `plots_d=60/`, `plots_d=60_AB/` — warianty wyników dla
  różnych szerokości złącza `d` (w nm) i konfiguracji geometrii (`AB` —
  konfiguracja Bernala).

Typowe nazewnictwo plików:

| Wzorzec | Zawartość |
|---|---|
| `BLG_sf{1,4}_dispersion.npy` | Struktura pasmowa $E(k)$ dla danego `s_f` |
| `zadanie{1,2,4,5,6}_*.npy/.npz` | Wyniki poszczególnych zadań |
| `zadanie8_G_map_*_full.npy` + `*_meta.npz` | Pełna mapa 2D $G(B, V_b)$ i jej metadane |
| `zadanie8_line_*.npz` | Przekrój $G(B)$ przy stałym $V_b$ |
| `task8_chunks*/task8_chunk_XXXX_START_END.npz` | Pojedynczy fragment mapy zadania 8 |
| `zadanie9_*/...` | Lokalny prąd wiązaniowy i metadane |
| `BLG_solver_params_*.npz` | Wyniki sweepu solvera samouzgodnionego |
| `trajectories_{l_to_r,r_to_l}/*.npz` | Trajektorie stanów snake |

## Uruchamianie na klastrze SLURM

Skrypty `run_*.sh` konfigurują środowisko PLGrid (moduły `GCC`,
`OpenMPI`, `Python`, `MUMPS`), aktywują wirtualne środowisko Python i
uruchamiają odpowiedni skrypt Python:

| Skrypt SLURM | Uruchamia | Uwagi |
|---|---|---|
| [run_BLG.sh](run_BLG.sh) | analizę / obliczenia główne | 1 węzeł, ~5 GB, 72h |
| [run_BLG_plots.sh](run_BLG_plots.sh) | `main_BLG_plots.py` | `MPLBACKEND=Agg` (bez GUI) |
| [run_BLG_task8_chunk.sh](run_BLG_task8_chunk.sh) | `main_BLG_task8_chunk.py` | wspiera array jobs (`SLURM_ARRAY_TASK_ID` → `CHUNK_INDEX`) |
| [run_merge_BLG_task8_chunks.sh](run_merge_BLG_task8_chunks.sh) | `merge_BLG_task8_chunks.py` | scalanie po zakończeniu wszystkich chunków |
| [run_BLG_task9.sh](run_BLG_task9.sh) | `main_BLG_task9.py` | ~32 GB RAM (macierz S) |
| [run_BLG_task_F.sh](run_BLG_task_F.sh) | `main_BLG_task_F.py` | ~8 GB RAM, 24h |
| [run_BLG_trajectories.sh](run_BLG_trajectories.sh) | `main_BLG_trajectories.py` | krótkie zadanie, ~4 GB, 1h |
| [submit_task9_jobs.sh](submit_task9_jobs.sh) / [submit_task9_jobs_fixed_B.sh](submit_task9_jobs_fixed_B.sh) | pętla `sbatch` po wielu wartościach $B$ dla zadania 9 | konfiguracja wartości `B_VALUES` w skrypcie |
| [submit_taskF_jobs.sh](submit_taskF_jobs.sh) | pętla `sbatch` po wartościach $B$ dla zadania F | ustalone `VT_VALUE`, `VB_VALUE` |

Typowy przepływ pracy:

1. **Obliczenia**: `sbatch run_BLG.sh` (lub, dla zadania 8, submit array
   jobs przez `run_BLG_task8_chunk.sh`, a następnie
   `sbatch run_merge_BLG_task8_chunks.sh` do scalenia wyników).
2. **Wykresy**: `sbatch run_BLG_plots.sh`.
3. **Analiza**: uruchomienie lokalne/na klastrze
   `analyze_task8_line_extrema.py` lub
   `analyze_task8_map_fixed_B_extrema.py`.

## Wymagania i instalacja

Projekt wymaga Pythona 3.11+ oraz następujących pakietów:

- [`kwant`](https://kwant-project.org/) — budowa układów sieciowych i
  obliczenia transportu kwantowego (macierz S),
- `numpy`, `scipy` — algebra liniowa, całkowanie (`quad`), szukanie
  pierwiastków (`fsolve`, `brentq`), ekstrema (`find_peaks`),
- `matplotlib` — wykresy (backend `Agg` na klastrze),
- `mpi4py` oraz `MUMPS` — opcjonalny solver sparse dla Kwant przy większych
  układach.

Instalacja (przykład lokalny):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install kwant numpy scipy matplotlib mpi4py
```

Na klastrze PLGrid środowisko jest aktywowane w skryptach `run_*.sh` przez
`source $SCRATCH/BLG_calc/bin/activate` po załadowaniu odpowiednich
modułów (`module load GCC/... OpenMPI/... Python/... MUMPS/...`).

## Konwencje w kodzie

- Wewnętrznie stosowane są **jednostki atomowe** ($\hbar = e = 1$);
  przeliczniki na jednostki SI/eV/nm/T znajdują się w
  [Constants.py](Constants.py).
- Nazwy funkcji/zadań są w większości w języku polskim (`zadanie`,
  `wykres`, `przewodnosc`, `dyspersja`), zgodnie z numeracją zadań pracy
  magisterskiej.
- Skrypty `main_BLG_data.py` i `main_BLG_plots.py` są rozdzielone
  celowo — obliczenia (kosztowne, na klastrze) są oddzielone od
  rysowania (szybkie, można iterować lokalnie na zapisanych danych).
- Duże obliczenia (zadanie 8 — mapa 2D $G(B, V_b)$) są dzielone na chunki
  po $B$, liczone niezależnie jako SLURM array jobs, a następnie scalane
  skryptem `merge_BLG_task8_chunks.py`.
- Współczynnik skalowania sieci `s_f` w `BLGSystemParameters` realizuje
  metodę "scalable tight-binding" — powiększenie stałej sieci przy
  zachowaniu niskoenergetycznej fizyki Diraca, co pozwala redukować koszt
  obliczeniowy dla większych układów.
