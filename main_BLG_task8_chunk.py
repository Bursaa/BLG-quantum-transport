#!/usr/bin/env python3
"""Uruchomienie zadania 8 w częściach po N punktów B."""

import argparse
import os
from pathlib import Path

import numpy as np

from main_BLG import *  # noqa: F401,F403
from BLG_transport import conductance_map_G_B_Vb, ensure_data_dir


L = 300.0 / l_scale
W = 150.0 / l_scale
d = 0.001 / l_scale


def parse_args():
    parser = argparse.ArgumentParser(description="Oblicz chunk zadania 8 dla mapy G(B, Vb)")
    parser.add_argument("--chunk-index", type=int, default=int(os.getenv("CHUNK_INDEX", "0")))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "5")))
    parser.add_argument("--b-start", type=float, default=float(os.getenv("B_START", "0.0")))
    parser.add_argument("--b-stop", type=float, default=float(os.getenv("B_STOP", "10.01")))
    parser.add_argument("--b-step", type=float, default=float(os.getenv("B_STEP", "0.1")))
    parser.add_argument("--vb-start", type=float, default=float(os.getenv("VB_START", "0.0")))
    parser.add_argument("--vb-stop", type=float, default=float(os.getenv("VB_STOP", "50.0")))
    parser.add_argument("--vb-step", type=float, default=float(os.getenv("VB_STEP", "0.1")))
    parser.add_argument("--vt", type=float, default=float(os.getenv("VT_VALUE", "0.0")))
    return parser.parse_args()


def make_range(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(np.floor((stop - start) / step + 1e-12)) + 1
    return np.arange(n) * step + start


def main():
    args = parse_args()
    ensure_data_dir()

    B_values = make_range(args.b_start, args.b_stop, args.b_step)
    Vb_arr = make_range(args.vb_start, args.vb_stop, args.vb_step)

    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")

    chunk_start = args.chunk_index * args.chunk_size
    chunk_end = min(chunk_start + args.chunk_size, len(B_values))

    if chunk_start >= len(B_values):
        print(f"Chunk {args.chunk_index} poza zakresem; brak danych do obliczenia.")
        return

    B_chunk = B_values[chunk_start:chunk_end]
    out_dir = Path("data_d=0_AB/task8_chunks_TEST")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"task8_chunk_{args.chunk_index:04d}_{chunk_start:04d}_{chunk_end:04d}.npz"

    print(f"Obliczam chunk {args.chunk_index}: B[{chunk_start}:{chunk_end}] / {len(B_values)}")
    print(f"  B_chunk = {B_chunk[:3]} ... {B_chunk[-1]}")
    print(f"  Vb_arr len = {len(Vb_arr)}")

    cap_params = CapacitanceParameters(Cm=2.55, Cb=0.653, Ct=2.55, Cg=460.0)
    phys_params = BLGPhysicsParameters(gamma1=0.39, hvf2=6.39**2, n_t=118.57)
    base_params = BLGSystemParameters(
        L=L,
        W=W,
        s_f=4.0,
        t=T_INTRALAYER,
        gamma1=GAMMA1,
        name=f"BLG_task8_chunk_{args.chunk_index}",
        d=d,
    )

    G_chunk = conductance_map_G_B_Vb(
        base_params,
        B_chunk,
        Vb_arr,
        Vt=args.vt,
        cap_params=cap_params,
        phys_params=phys_params,
        uniform_B=False,
        verbose=True,
    )

    np.savez(
        out_file,
        G=G_chunk,
        B=B_chunk,
        Vb=Vb_arr,
        Vt=np.array([args.vt]),
        chunk_index=np.array([args.chunk_index]),
        chunk_size=np.array([args.chunk_size]),
        start_idx=np.array([chunk_start]),
        end_idx=np.array([chunk_end]),
    )

    print(f"Zapisano: {out_file}")


if __name__ == "__main__":
    main()
