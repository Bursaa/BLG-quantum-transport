"""Przerysuj istniejace trajektorie (data_d=0_AB, data_d=60_AB) z wieksza legenda i nowym tytulem."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_ROOTS = ["data_d=0_AB", "data_d=60_AB"]
DIRECTIONS = ["trajectories_l_to_r", "trajectories_r_to_l"]
X_LIMIT_NM = 200.0
Y_LIMIT_NM = 150.0

# Ogranicz zestaw B rysowanych dla wybranych Vb (klucz -> dozwolone wartosci B_T)
B_VALUES_BY_VB = {
    30.0: (0.48, 0.7, 1.3, 2.18, 4.98, 5.75, 6.3, 6.8, 7.8, 8.85),
}


def replot_folder(vb_dir: Path):
    traj_files = sorted(vb_dir.glob("trajectory_B*.npz"))
    if not traj_files:
        return

    records = []
    for f in traj_files:
        with np.load(f) as d:
            records.append((float(d["B_T"]), d["x_nm"], d["y_nm"],
                             float(d["Vt"]), float(d["Vb"])))
    records.sort(key=lambda r: r[0])

    Vb_check = records[0][4]
    allowed = B_VALUES_BY_VB.get(Vb_check)
    if allowed is not None:
        records = [r for r in records
                   if any(abs(r[0] - b) < 1e-6 for b in allowed)]
    if not records:
        return

    Vt = records[0][3]
    Vb = records[0][4]

    fig, ax = plt.subplots(figsize=(10, 7))
    for B_abs, xs, ys, _, _ in records:
        ax.plot(xs, ys, lw=1.4, label=f"B={B_abs:.2f} T")

    ax.axvline(0.0, color="black", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlim(-X_LIMIT_NM, X_LIMIT_NM)
    ax.set_ylim(-Y_LIMIT_NM, Y_LIMIT_NM)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"Trajektorie semiklasyczne: Vt={Vt:.0f} V, Vb={Vb:.0f} V")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=13, ncol=2)
    fig.tight_layout()
    fig.savefig(vb_dir / "trajectories_all.png", dpi=180)
    fig.savefig(vb_dir / "trajectories_all.pdf")
    plt.close(fig)
    print(f"Zapisano: {vb_dir}")


def main():
    for root in DATA_ROOTS:
        for direction in DIRECTIONS:
            base = Path(root) / direction
            if not base.is_dir():
                continue
            for vb_dir in sorted(base.iterdir()):
                if vb_dir.is_dir():
                    replot_folder(vb_dir)


if __name__ == "__main__":
    main()
