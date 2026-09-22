#!/usr/bin/env python3
"""Standalone analysis for a fixed-B slice of the task8 full map.

Loads the full map G(Vb, B) from:
    data_d=0_AB/zadanie8_G_map_Vt0_full.npy
and meta from:
    data_d=0_AB/zadanie8_G_map_meta_Vt0_full.npz

Then for each requested magnetic field B it:
1) extracts the G(Vb) slice at the nearest B,
2) denoises the slice with a 5% low-pass FFT filter,
3) finds all local maxima and minima,
4) plots the cleaned curve with extrema markers,
5) plots the FFT amplitude in log scale,
6) prints the corresponding (Vb, G) values.

Run examples:
    python analyze_task8_map_fixed_B_extrema.py --B 1.0
    python analyze_task8_map_fixed_B_extrema.py --B 1.0 2.0 5.0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import re

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

MAP_FILE = Path("data_d=0_AB") / "zadanie8_G_map_Vt0_full.npy"
META_FILE = Path("data_d=0_AB") / "zadanie8_G_map_meta_Vt0_full.npz"
PLOT_DIR = Path("plots_d=0_AB")
LOWPASS_FRACTION = 0.1


def smooth_curve(y: np.ndarray, retain_fraction: float | None = None) -> np.ndarray:
    """Keep only the lowest frequencies in the Fourier spectrum."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return y.copy()

    if retain_fraction is None:
        retain_fraction = LOWPASS_FRACTION

    spectrum = np.fft.rfft(y)
    kept_bins = max(1, int(retain_fraction * spectrum.size))
    mask = np.zeros_like(spectrum, dtype=bool)
    mask[:kept_bins] = True

    filtered = np.fft.irfft(spectrum * mask, n=y.size)
    return filtered


def find_extrema(y_smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find local maxima and minima while suppressing tiny ripples."""
    if y_smooth.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    spacing = max(5, int(y_smooth.size * 0.02))
    prominence = max(0.05, 0.08 * np.ptp(y_smooth))

    maxima, _ = find_peaks(y_smooth, distance=spacing, prominence=prominence)
    minima, _ = find_peaks(-y_smooth, distance=spacing, prominence=prominence)
    return maxima, minima


def load_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not MAP_FILE.exists():
        raise FileNotFoundError(f"Missing map: {MAP_FILE}")
    if not META_FILE.exists():
        raise FileNotFoundError(f"Missing metadata: {META_FILE}")

    G_map = np.asarray(np.load(MAP_FILE), dtype=float)
    meta = np.load(META_FILE)
    B = np.asarray(meta["B"], dtype=float).ravel()
    Vb = np.asarray(meta["Vb"], dtype=float).ravel()

    if G_map.shape != (B.size, Vb.size):
        raise ValueError(
            f"Unexpected map shape {G_map.shape}; expected {(B.size, Vb.size)} "
            f"from B={B.size}, Vb={Vb.size}."
        )

    return B, Vb, G_map


def extract_slice_at_B(B_target: float, B: np.ndarray, Vb: np.ndarray, G_map: np.ndarray):
    idx_B = int(np.argmin(np.abs(B - B_target)))
    B_actual = B[idx_B]
    G_slice = np.asarray(G_map[idx_B, :], dtype=float)
    return idx_B, B_actual, Vb, G_slice


def print_extrema(label: str, Vb: np.ndarray, G: np.ndarray, maxima: np.ndarray, minima: np.ndarray) -> None:
    print(f"\n=== {label} ===")
    print(f"Liczba maksimów: {maxima.size}")
    print(f"Liczba minimów:  {minima.size}")

    if maxima.size:
        print("Maksima:")
        for idx in maxima:
            print(f"  Vb = {Vb[idx]:.6f} V, G = {G[idx]:.8f}")
    else:
        print("Maksima: brak")

    if minima.size:
        print("Minima:")
        for idx in minima:
            print(f"  Vb = {Vb[idx]:.6f} V, G = {G[idx]:.8f}")
    else:
        print("Minima: brak")


def plot_extrema(label: str, Vb: np.ndarray, G_raw: np.ndarray, G_smooth: np.ndarray, maxima: np.ndarray, minima: np.ndarray) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Vb, G_raw, color="tab:blue", linewidth=1.5, alpha=0.7, label="surowe G(Vb)")
    ax.plot(Vb, G_smooth, color="tab:red", linewidth=2.0, label="składowa wolnozmienna G(Vb)")

    if maxima.size:
        ax.scatter(Vb[maxima], G_smooth[maxima], color="green", s=70, label="maksima", zorder=5)
    if minima.size:
        ax.scatter(Vb[minima], G_smooth[minima], color="red", s=70, label="minima", zorder=5)
    b_match = re.search(r"B_([\d.]+)_", label)
    B = int(float(b_match.group(1))) if b_match else label
    ax.set_title(rf"$B = {B}$T: minima i maksima po filtracji")
    ax.set_xlabel("Vb (V)")
    ax.set_ylabel("G (2e^2/h)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    out_file = PLOT_DIR / f"{label}_extrema.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Zapisano wykres: {out_file}")


def plot_frequency_spectrum(label: str, Vb: np.ndarray, G_raw: np.ndarray, G_smooth: np.ndarray) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if Vb.size < 2:
        return

    dVb = np.median(np.diff(Vb))
    if not np.isfinite(dVb) or dVb <= 0:
        return

    freqs = np.fft.rfftfreq(Vb.size, d=dVb)
    amplitude_raw = np.abs(np.fft.rfft(G_raw))
    cutoff_bin = max(1, int(LOWPASS_FRACTION * freqs.size))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogy(freqs, amplitude_raw, color="tab:blue", linewidth=1.7, label="FFT oryginalna")
    ax.axvline(freqs[cutoff_bin - 1], color="gray", linestyle="--", linewidth=1.2,
               label=f"cutoff {LOWPASS_FRACTION * 100:.0f}% (f ≈ {freqs[cutoff_bin - 1]:.4f} 1/V)")
    b_match = re.search(r"B_([\d.]+)_", label)
    B = int(float(b_match.group(1))) if b_match else label
    ax.set_title(rf"$B = {B}$T: widmo w osi częstotliwości")
    ax.set_xlabel("częstotliwość f (1/V)")
    ax.set_ylabel("|FFT(G(Vb))| (log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    out_file = PLOT_DIR / f"{label}_frequency_spectrum.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Zapisano wykres widma: {out_file}")


def main() -> None:
    B_VALUES_TO_ANALYZE = [9.0]

    B_all, Vb_all, G_map = load_map()

    for B_target in sorted(set(B_VALUES_TO_ANALYZE)):
        idx_B, B_actual, Vb, G_slice = extract_slice_at_B(B_target, B_all, Vb_all, G_map)
        G_smooth = smooth_curve(G_slice)
        maxima, minima = find_extrema(G_smooth)

        label = f"zadanie8_map_B_{B_actual:.3f}_T"
        print(f"\nB_target={B_target:.3f} T -> nearest B={B_actual:.6f} T (index {idx_B})")
        print_extrema(label, Vb, G_smooth, maxima, minima)
        plot_extrema(label, Vb, G_slice, G_smooth, maxima, minima)
        plot_frequency_spectrum(label, Vb, G_slice, G_smooth)


if __name__ == "__main__":
    main()
