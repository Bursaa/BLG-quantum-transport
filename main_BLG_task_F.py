#!/usr/bin/env python3
"""Obliczenie dyspersji (wykres F) dla jednej wartości B."""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Wykres F: dyspersja układu dla jednej wartości B")
    parser.add_argument("--B-T",  type=float, default=float(os.getenv("B_T",       "1.0")))
    parser.add_argument("--Vt",   type=float, default=float(os.getenv("VT_VALUE",  "0.0")))
    parser.add_argument("--Vb",   type=float, default=float(os.getenv("VB_VALUE", "30.0")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Wykres F: B_T={args.B_T} T, Vt={args.Vt} V, Vb={args.Vb} V", flush=True)
    from main_BLG_data import zadanie_F_dyspersja
    zadanie_F_dyspersja(Vt=args.Vt, Vb=args.Vb, B_T=args.B_T)
