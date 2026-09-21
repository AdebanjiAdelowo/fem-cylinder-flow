# Benchmark definition audit

Every quantity of the DFG 2D-1 / 2D-2 benchmark, its definition, where it is implemented, and how
it is protected by a test. Definitions were checked against the benchmark maintainers' pages
(FeatFlow, "DFG benchmark 2D-1 (RE20, laminar)" and "DFG benchmark 2D-2 (RE100, periodic)", read
2026-09-21; the 2D-2 page's time series were also downloaded and analysed, see
`src/reference.py`). The original 1996 paper (Schaefer and Turek) was **not** re-read: the
comparison intervals in `src/reference.py` are carried over from the repository's first commit and
are marked as such there.

| Item | Benchmark definition | Implementation | Status |
|---|---|---|---|
| Channel | [0, 2.2] x [0, 0.41] | `src/geometry.py` `L`, `H` | matches; `test_geometry_constants_match_schafer_turek_1996`, `test_domain_area_is_channel_minus_cylinder` |
| Cylinder | centre (0.2, 0.2), diameter 0.1 | `CX, CY, R = 0.2, 0.2, 0.05` | matches; the meshed boundary is checked to lie on that circle (`test_curved_boundary_error_is_orders_of_magnitude_below_polygon`, `test_wall_and_cylinder_facets_sit_at_the_specified_places`) |
| Fluid | nu = 1e-3, rho = 1 | `NU`, `RHO` in the run scripts | matches; `test_benchmark_normalisation_and_reynolds_numbers` |
| Inflow | u(0, y) = 4 U y (0.41 - y) / 0.41^2, U = 0.3 (2D-1), 1.5 (2D-2), time independent | `src/cylinder_bcs.py` | matches; `test_inflow_profile_matches_benchmark_formula` |
| Reynolds number | Re = Ubar D / nu, Ubar = (2/3) U | `U_BAR = 2/3 * U_M` | Re = 20 and 100; tested |
| Walls, cylinder | no slip | `build_bcs` | matches |
| Outflow | do nothing: nu d_eta u - p eta = 0 | natural condition of the weak form `nu grad(u):grad(v) - p div v` (no outlet term) | matches, and the Laplacian viscous form is what makes this the stated condition |
| Initial condition | zero solution | `w_prev = 0` (2D-2 from rest); restarts start from a saved developed state | see "Differences" |
| Pressure probes | a1 = (0.15, 0.2), a2 = (0.25, 0.2); p_diff = p(a1) - p(a2) | `BENCHMARK_PRESSURE_POINTS`, `PointProbe` | matches; both points lie exactly on the cylinder surface, so the probe falls back to the nearest cell if a curved boundary misses the point by a hair (`test_probe_*`) |
| Force | (F_D, F_L) = int_S sigma eta ds, sigma = nu grad(u) - p I, eta = outer normal of the circle | method `laplacian` in `src/forces.py`, F = -int sigma n ds with n the fluid outward normal (eta = -n) | matches. The repository originally used `symmetric`, sigma = -pI + nu (grad u + grad u^T), which is identical only for div u = 0 and differs at O(h) for Taylor-Hood (see below) |
| Alternative force formula | F_D = int (nu d_eta u_t eta_y - p eta_x), F_L = -int (nu d_eta u_t eta_x + p eta_y) | method `tangential` | implemented for comparison |
| Coefficients | c = 2 F / (Ubar^2 L), L = 0.1 | `ForceCoefficients.norm = 0.5 rho Ubar^2 D` | matches; sign, normal direction and Ubar-scaling verified against exact divergence-theorem integrals (`test_pressure_gradient_force_*`, `test_viscous_force_*`) |
| Strouhal number | St = L f / Ubar | `src/periodic.py`; headline value from the mean c_L-minimum-to-minimum period over the last 4 cycles, cross-checked by a zero-padded windowed FFT and a harmonic least-squares fit | matches; `test_frequency_estimators_agree_on_off_bin_frequency` |
| Cycle | from a minimum of c_L to the next | `find_cycles` | matches |
| Extrema | max / min / mean / amp of c_D, c_L over a cycle | cubic-spline extrema per cycle | see below |
| Measurement window | fully developed flow, an arbitrary cycle between 25 s and 30 s | last cycle of the run, judged periodic by cycle-to-cycle drift | see "Differences" |

## Differences from the benchmark procedure, stated explicitly

* **Stress convention.** The repository's original force used the symmetric stress. On the steady
  2D-1 solution at 61k dofs the four force definitions differ by 0.5 percent (polygon) in c_D,
  which is large next to the 0.2 to 0.6 percent effects studied here. All four are now computed
  and reported; they converge to the same value as h -> 0.
* **Measurement window.** The benchmark procedure integrates about 25 s and measures in 25 to 30 s.
  The runs here stop at 8 s (from rest) or 12 s (continued from a developed state). The periodic
  regime is therefore not assumed: it is judged from the drift between the last two lift cycles
  (`analyse_periodic`), and the reported values are those of the last complete cycle. The
  cycle-to-cycle drift at the end of the 8 s runs is about 3e-5 in c_L,max and 1e-5 in c_D,max.
* **Impulsive start.** The inflow is switched on in one step, as in the benchmark's "start with a
  zero solution", but the coarse-step start-up phase prescribed by the benchmark is not used.
* **Pressure difference.** The FeatFlow page defines p_diff as a time series over one cycle and gives
  no single-instant value. The 1996 paper's table may define a specific instant (for example half
  a period after the lift maximum); that could not be checked here, so the pressure difference is
  reported at the lift maximum and half a period later. They differ by about 0.002.
* **Time level of the forces.** The traction forms use (u, p) at t_{n+1}. The pressure of the
  Crank-Nicolson scheme is a multiplier for the time-averaged momentum equation, so it
  approximates t_{n+1/2}. The `variational` force is centred at t_{n+1/2} by construction.
