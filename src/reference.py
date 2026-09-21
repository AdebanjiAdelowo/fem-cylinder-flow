"""Published reference data used ONLY to report where a computed benchmark quantity lies.

Nothing in this module feeds back into the solver or the tests; it exists so that the comparison
values, and where each came from, are stated in one place.

Provenance
----------
``INTERVALS_*``  the "lower/upper bound" rows of Tables 3 and 4 of Schaefer & Turek (1996). They have
                 been the repository's comparison intervals since its first commit. The paper itself
                 was NOT re-read while the convergence study was carried out, so these numbers are
                 carried over, not independently re-verified.
``SPECTRAL_2D1`` the high-order spectral reference values for 2D-1 (Nabh 1998) published on the
                 FeatFlow benchmark page, https://www.mathematik.tu-dortmund.de/~featflow/en/
                 benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html (read 2026-09-21).
``FEATFLOW_2D2`` per-refinement-level 2D-2 values computed HERE from the time series distributed
                 with the FeatFlow page (20141124_dfg_bench2_official.zip, "new reference results",
                 Q2/P1disc, Crank-Nicolson, higher-order line integration of the forces), using the
                 same cubic-spline cycle extraction as ``src.periodic`` on the 1 s window t in
                 [9, 10] s. Level 6 (667,264 dofs) at dt = 1/1600 is the finest published entry.
                 These are the maintainers' own numerical results, not certified bounds.
"""
from __future__ import annotations

INTERVALS_2D1 = {"c_D": (5.5700, 5.5900), "c_L": (0.0104, 0.0110), "dP": (0.1172, 0.1176)}
INTERVALS_2D2 = {"cd_max": (3.2200, 3.2400), "cl_max": (0.9900, 1.0100),
                 "st": (0.2950, 0.3050), "dP": (2.4600, 2.5000)}

SPECTRAL_2D1 = {"c_D": 5.57953523384, "c_L": 0.010618948146, "dP": 0.11752016697}

# level -> {dt: (cd_max, cl_max, period, dP at c_L,max)}; extracted with ref_analysis (see docstring)
FEATFLOW_2D2 = {
    3: {"1/100": (3.20943, 0.97944, 0.33228, 2.4954), "1/400": (3.20945, 0.98288, 0.33121, 2.4953),
        "1/1600": (3.20945, 0.98309, 0.33114, 2.4953)},
    4: {"1/100": (3.22327, 0.98123, 0.33234, 2.4851), "1/400": (3.22329, 0.98465, 0.33127, 2.4850),
        "1/1600": (3.22329, 0.98486, 0.33120, 2.4850)},
    5: {"1/100": (3.22661, 0.98257, 0.33241, 2.4832), "1/400": (3.22662, 0.98599, 0.33135, 2.4831),
        "1/1600": (3.22662, 0.98620, 0.33128, 2.4831)},
    6: {"1/100": (3.22738, 0.98295, 0.33243, 2.4830), "1/400": (3.22739, 0.98637, 0.33137, 2.4829),
        "1/1600": (3.22739, 0.98658, 0.33130, 2.4829)},
}


def interval_status(value: float, interval: tuple[float, float]) -> str:
    lo, hi = interval
    if lo <= value <= hi:
        return "inside"
    nearest = lo if value < lo else hi
    return f"outside by {100.0 * (value - nearest) / nearest:+.2f}% (vs nearest bound {nearest})"
