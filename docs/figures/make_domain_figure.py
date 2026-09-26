"""Draw the DFG 2D-1/2D-2 computational domain and boundary conditions.

Geometry constants and pressure-probe locations are parsed from src/geometry.py and
src/forces.py (without importing dolfinx), so the figure always matches the solver.

    python docs/figures/make_domain_figure.py   ->   docs/figures/domain_bcs.svg
"""
import ast
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
geo = (ROOT / "src" / "geometry.py").read_text()
L, H = (float(v) for v in re.search(r"^L, H = (.+)$", geo, re.M).group(1).split(","))
CX, CY, R = (float(v) for v in re.search(r"^CX, CY, R = (.+)$", geo, re.M).group(1).split(","))
probes = ast.literal_eval(
    re.search(r"^BENCHMARK_PRESSURE_POINTS = (.+)$",
              (ROOT / "src" / "forces.py").read_text(), re.M).group(1))

plt.rcParams.update({"font.size": 9, "svg.fonttype": "path", "svg.hashsalt": "fig"})
fig, ax = plt.subplots(figsize=(10, 2.9))
fig.patch.set_facecolor("white")

channel = plt.Rectangle((0, 0), L, H, fc="#eef3f8", ec="none")
ax.add_patch(channel)
# refinement band of the gmsh Threshold field (SizeMin within 2R, SizeMax beyond 8R),
# clipped to the fluid domain
for rad, fc in ((8 * R, "#dce8f3"), (2 * R, "#bcd3e8")):
    c = ax.add_patch(Circle((CX, CY), rad, fc=fc, ec="#9db7cf", lw=0.6, ls="--"))
    c.set_clip_path(channel)
ax.add_patch(Circle((CX, CY), R, fc="white", ec="#1f3b57", lw=1.6))

# walls, inlet, outlet
ax.plot([0, L], [0, 0], color="#1f3b57", lw=2.2)
ax.plot([0, L], [H, H], color="#1f3b57", lw=2.2)
ax.plot([0, 0], [0, H], color="#c0392b", lw=2.2)
ax.plot([L, L], [0, H], color="#27864a", lw=2.2)

# parabolic inflow profile, u(0, y) = 4 U_m y (H - y) / H^2 (drawn normalised)
y = np.linspace(0, H, 9)[1:-1]
prof = 4 * y * (H - y) / H**2
for yi, pi in zip(y, prof):
    ax.add_patch(FancyArrowPatch((0.0, yi), (0.16 * pi, yi), arrowstyle="-|>",
                                 mutation_scale=7, color="#c0392b", lw=0.9))
yy = np.linspace(0, H, 100)
ax.plot(0.16 * 4 * yy * (H - yy) / H**2, yy, color="#c0392b", lw=0.8)

for (px, py) in probes:
    ax.plot(px, py, "o", ms=4, color="#e67e22", zorder=5)
ax.annotate("pressure probes (front, rear)\n$\\Delta P = p(0.15,0.2) - p(0.25,0.2)$",
            xy=probes[1], xytext=(0.66, 0.30), fontsize=8, color="#a04e00",
            arrowprops=dict(arrowstyle="-", color="#a04e00", lw=0.7))
ax.annotate("cylinder $\\Gamma_4$: $\\mathbf{u}=0$\ndrag and lift integrated here",
            xy=(CX + R * np.cos(-0.8), CY + R * np.sin(-0.8)), xytext=(0.66, 0.045),
            fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.7))
ax.annotate("mesh refined near the cylinder\n(size $h_{cyl}$ within $2R$, "
            "graded to $h_{far}$ by $8R$)", xy=(CX + 0.385, CY + 0.06), xytext=(1.25, 0.20),
            fontsize=8, color="#3d5a78", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#3d5a78"))

ax.text(-0.03, H / 2, "inlet $\\Gamma_1$\n$u = 4U_m\\,y(H-y)/H^2$\n$v = 0$", ha="right",
        va="center", color="#c0392b", fontsize=8)
ax.text(L + 0.03, H / 2, "outlet $\\Gamma_2$\ndo-nothing\n(natural BC)", ha="left",
        va="center", color="#27864a", fontsize=8)
ax.text(L / 2 + 0.45, H + 0.018, "wall $\\Gamma_3$: $\\mathbf{u} = 0$", ha="center",
        va="bottom", color="#1f3b57", fontsize=8)
ax.text(L / 2 + 0.45, -0.018, "wall $\\Gamma_3$: $\\mathbf{u} = 0$", ha="center", va="top",
        color="#1f3b57", fontsize=8)

# dimensions
ax.annotate("", xy=(0, -0.085), xytext=(L, -0.085),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35"))
ax.text(L / 2, -0.098, f"L = {L} m", ha="center", va="top", fontsize=8, color="0.35")
ax.annotate(f"cylinder: D = {2 * R:g} m,\ncentre ({CX:g}, {CY:g})", xy=(CX, CY + R),
            xytext=(0.30, H + 0.035), fontsize=7.5, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("", xy=(L + 0.33, 0), xytext=(L + 0.33, H),
            arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35"))
ax.text(L + 0.35, H / 2 - 0.12, f"H = {H} m", rotation=90, va="center", fontsize=8, color="0.35")

ax.set_xlim(-0.45, L + 0.45)
ax.set_ylim(-0.16, H + 0.11)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
out = Path(__file__).with_name("domain_bcs.svg")
fig.savefig(out, facecolor="white", metadata={"Date": None})  # deterministic output
if len(sys.argv) > 1:  # optional raster preview path
    fig.savefig(sys.argv[1], dpi=150, facecolor="white")
print(f"wrote {out.relative_to(ROOT)}")
