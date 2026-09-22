#!/usr/bin/env python3
"""Scala wyniki chunków zadania 8 do jednej mapy G(B, Vb)."""

from pathlib import Path

import numpy as np


def main():
    chunk_dir = Path("data_d=0_AB/task8_chunks")
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Brak katalogu z chunkami: {chunk_dir}")

    chunk_files = sorted(chunk_dir.glob("task8_chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError("Brak plików chunków do scalenia")

    row_by_B = {}
    B_by_key = {}
    Vb_ref = None
    vt_values = set()
    duplicates = 0

    for path in chunk_files:
        with np.load(path) as data:
            B = np.asarray(data["B"], dtype=float).ravel()
            G = np.asarray(data["G"], dtype=float)
            Vb = np.asarray(data["Vb"], dtype=float).ravel()
            vt = float(np.asarray(data.get("Vt", np.array([0.0]))).ravel()[0])

            if G.ndim != 2:
                raise ValueError(f"{path}: oczekiwano G o wymiarze 2D, otrzymano {G.shape}")
            if G.shape[0] != len(B):
                raise ValueError(
                    f"{path}: G.shape[0]={G.shape[0]} nie zgadza się z len(B)={len(B)}"
                )
            if G.shape[1] != len(Vb):
                raise ValueError(
                    f"{path}: G.shape[1]={G.shape[1]} nie zgadza się z len(Vb)={len(Vb)}"
                )

            if Vb_ref is None:
                Vb_ref = Vb
            elif len(Vb) != len(Vb_ref):
                if len(Vb) == len(Vb_ref) + 1 and np.allclose(Vb[:-1], Vb_ref):
                    Vb = Vb_ref
                elif len(Vb_ref) == len(Vb) + 1 and np.allclose(Vb, Vb_ref[:-1]):
                    Vb = Vb_ref
                else:
                    raise ValueError(f"{path}: niespójna siatka Vb między chunkami")
            elif not np.allclose(Vb_ref, Vb):
                if np.allclose(Vb[:-1], Vb_ref[:-1]):
                    Vb = Vb_ref
                elif np.allclose(Vb[1:], Vb_ref[1:]):
                    Vb = Vb_ref
                else:
                    raise ValueError(f"{path}: niespójna siatka Vb między chunkami")

            vt_values.add(vt)

            for i, b_val in enumerate(B):
                key = round(float(b_val), 12)
                if key in row_by_B:
                    duplicates += 1
                row_by_B[key] = G[i, :]
                B_by_key[key] = float(b_val)

    if Vb_ref is None or not row_by_B:
        raise RuntimeError("Nie udało się odczytać żadnych wierszy danych z chunków")

    sorted_keys = sorted(row_by_B.keys())
    B_all = np.array([B_by_key[k] for k in sorted_keys], dtype=float)
    G_all = np.vstack([row_by_B[k] for k in sorted_keys])

    out_dir = Path("data_d=0_AB")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / "zadanie8_G_map_Vt0_full.npy"
    meta_file = out_dir / "zadanie8_G_map_meta_Vt0_full.npz"

    vt_out = float(next(iter(vt_values))) if len(vt_values) == 1 else np.nan
    np.save(out_file, G_all)
    np.savez(meta_file, B=B_all, Vb=Vb_ref, Vt=np.array([vt_out]))

    print(f"Zapisano pełną mapę do {out_file}")
    print(f"Zapisano metadane do {meta_file}")
    print(f"Shape: {G_all.shape}")
    print(f"Liczba chunków: {len(chunk_files)}")
    print(f"Unikalne wiersze B: {len(B_all)}")
    if duplicates:
        print(f"Uwaga: nadpisano {duplicates} zduplikowanych wierszy B")
    if len(vt_values) > 1:
        print(f"Uwaga: wykryto różne Vt w chunkach: {sorted(vt_values)}")


if __name__ == "__main__":
    main()
