#!/usr/bin/env python3
"""Standalone analysis for task8 line data.

Reads the cached G(B) line files for Vb = 20, 30, 45 V and:
1) denoises each curve,
2) finds all local maxima and minima,
3) plots the cleaned curve with extrema markers,
4) prints the corresponding (B, G) values.

Run:
    /net/scratch/hscra/plgrid/plgbrodaczfilip/BLG_calc/bin/python analyze_task8_line_extrema.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


DATA_DIR = Path("data_d=60_AB")
PLOT_DIR = Path("plots_d=60_AB")
TARGET_VB = [20.0, 30.0, 45.0]


def smooth_curve(y: np.ndarray) -> np.ndarray:
    """Keep only the lowest 10% Fourier frequencies to suppress high-frequency noise."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return y.copy()

    spectrum = np.fft.rfft(y)
    kept_bins = max(1, int(0.10 * spectrum.size))
    mask = np.zeros_like(spectrum, dtype=bool)
    mask[:kept_bins] = True

    filtered = np.fft.irfft(spectrum * mask, n=y.size)
    return filtered


def find_extrema(y_smooth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reject tiny ripples by requiring a clear prominence and minimum spacing between extrema."""
    if y_smooth.size < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    spacing = max(5, int(y_smooth.size * 0.02))
    prominence = max(0.05, 0.08 * np.ptp(y_smooth))

    maxima, _ = find_peaks(y_smooth, distance=spacing, prominence=prominence)
    minima, _ = find_peaks(-y_smooth, distance=spacing, prominence=prominence)
    return maxima, minima


def load_line_data(vb: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    file_path = DATA_DIR / f"zadanie8_line_Vb_{vb}_Vt_0.0.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    data = np.load(file_path)
    B = np.asarray(data["B"], dtype=float).ravel()
    G = np.asarray(data["G"], dtype=float).ravel()

    if B.size != G.size:
        raise ValueError(
            f"Inconsistent lengths in {file_path}: B={B.size}, G={G.size}."
        )

    return B, G, np.asarray([vb], dtype=float)


def print_extrema(label: str, B: np.ndarray, G: np.ndarray, maxima: np.ndarray, minima: np.ndarray) -> None:
    print(f"\n=== {label} ===")
    print(f"Liczba maksimów: {maxima.size}")
    print(f"Liczba minimów:  {minima.size}")

    if maxima.size:
        print("Maksima:")
        for idx in maxima:
            print(f"  B = {B[idx]:.6f} T, G = {G[idx]:.8f}")
    else:
        print("Maksima: brak")

    if minima.size:
        print("Minima:")
        for idx in minima:
            print(f"  B = {B[idx]:.6f} T, G = {G[idx]:.8f}")
    else:
        print("Minima: brak")


def plot_extrema(label: str, B: np.ndarray, G_raw: np.ndarray, G_smooth: np.ndarray, maxima: np.ndarray, minima: np.ndarray) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(B, G_raw, color="tab:blue", linewidth=1.5, alpha=0.7, label="surowe G(B)")
    ax.plot(B, G_smooth, color="tab:red", linewidth=2.0, label="składowa wolnozmienna G(B)")

    if maxima.size:
        ax.scatter(B[maxima], G_smooth[maxima], color="green", s=70, label="maksima", zorder=5)
    if minima.size:
        ax.scatter(B[minima], G_smooth[minima], color="red", s=70, label="minima", zorder=5)
    vb_match = re.search(r"Vb_([\d.]+)_", label)
    Vb = int(float(vb_match.group(1))) if vb_match else label
    ax.set_title(rf"$V_{{b}} = {Vb}$ V: minima i maksima po filtracji")
    ax.set_xlabel("B (T)")
    ax.set_ylabel("G (2e^2/h)")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    out_file = PLOT_DIR / f"{label}_extrema.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Zapisano wykres: {out_file}")


def plot_frequency_spectrum(label: str, B: np.ndarray, G_raw: np.ndarray, G_smooth: np.ndarray) -> None:
    """Wykres widma amplitudowego dla oryginalnego sygnału z linią cutoff."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if B.size < 2:
        return

    dB = np.median(np.diff(B))
    if not np.isfinite(dB) or dB <= 0:
        return

    freqs = np.fft.rfftfreq(B.size, d=dB)
    amplitude = np.abs(np.fft.rfft(G_raw))
    cutoff_bin = max(1, int(0.10 * freqs.size))
    vb_match = re.search(r"Vb_([\d.]+)_", label)
    Vb = int(float(vb_match.group(1))) if vb_match else label
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.semilogy(freqs, amplitude, color="tab:purple", linewidth=1.7, label="FFT oryginalna")
    ax.axvline(freqs[cutoff_bin - 1], color="gray", linestyle="--", linewidth=1.2,
               label=f"cutoff 10% (f ≈ {freqs[cutoff_bin - 1]:.4f} 1/T)")
    ax.set_title(rf"$V_{{b}} = {Vb}$ V: widmo w osi częstotliwości")
    ax.set_xlabel("częstotliwość f (1/T)")
    ax.set_ylabel("|FFT(G(B))| (log)")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    out_file = PLOT_DIR / f"{label}_frequency_spectrum.png"
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    print(f"Zapisano wykres widma: {out_file}")


def main() -> None:
    for vb in TARGET_VB:
        B, G_raw, _ = load_line_data(vb)
        G_smooth = smooth_curve(G_raw)
        maxima, minima = find_extrema(G_smooth)

        label = f"zadanie8_line_Vb_{vb}_Vt_0_0"
        print_extrema(label, B, G_smooth, maxima, minima)
        plot_extrema(label, B, G_raw, G_smooth, maxima, minima)
        plot_frequency_spectrum(label, B, G_raw, G_smooth)


if __name__ == "__main__":
    main()
