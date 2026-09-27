"""Plot the drag and lift histories of the finest 2D-2 run (curved cylinder, L5 mesh).

Reads only the tracked time series results/2d2_timeseries_sp_L5.npz; nothing is re-simulated.
The benchmark-stress (`laplacian`) and `variational` forces are drawn, with the inherited
comparison intervals from src/reference.py shaded.

    python docs/figures/make_shedding_figure.py   ->   docs/figures/shedding_forces_L5.png
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
d = np.load(ROOT / "results" / "2d2_timeseries_sp_L5.npz")
ref = (ROOT / "src" / "reference.py").read_text()


def interval(name):
    """Pull a (lo, hi) tuple for `name` out of src/reference.py without importing dolfinx."""
    m = re.search(rf"[\"']{name}[\"']\s*:\s*\(([^)]+)\)", ref)
    return tuple(float(v) for v in m.group(1).split(","))


t = d["t"]
iv_d, iv_l = interval("cd_max"), interval("cl_max")
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(10, 4.8))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75])
ax_d = fig.add_subplot(gs[0, :2])
ax_l = fig.add_subplot(gs[1, :2], sharex=ax_d)
ax_z = fig.add_subplot(gs[:, 2])

for ax, key, label, iv in [(ax_d, "c_D", r"$c_D$", iv_d), (ax_l, "c_L", r"$c_L$", iv_l),
                           (ax_z, "c_L", r"$c_L$ (zoom on the last maximum)", iv_l)]:
    ax.axhspan(*iv, color="0.85", lw=0, label="inherited interval for the maximum")
    ax.plot(t, d[f"{key}_laplacian"], lw=1.1, label="benchmark stress (laplacian)")
    ax.plot(t, d[f"{key}_variational"], lw=1.1, ls="--", label="variational reaction")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)

ax_d.set_ylim(3.155, 3.245)
ax_d.tick_params(labelbottom=False)
ax_l.set_xlabel("t [s]")
ax_l.set_xlim(t[0], t[-1])

# zoom window: +/- 0.05 s around the last lift maximum
j = np.argmax(np.where(t > t[-1] - 0.34, d["c_L_laplacian"], -np.inf))
ax_z.set_xlim(t[j] - 0.05, t[j] + 0.05)
ax_z.set_ylim(0.955, 0.995)
ax_z.set_xlabel("t [s]")

h, lab = ax_d.get_legend_handles_labels()
fig.legend(h, lab, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.94))
fig.suptitle("DFG 2D-2, Re = 100: curved cylinder, 273,224 dofs, $\\Delta t = 0.0025$", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = Path(__file__).with_name("shedding_forces_L5.png")
fig.savefig(out, dpi=150)
print(f"wrote {out.relative_to(ROOT)}")
