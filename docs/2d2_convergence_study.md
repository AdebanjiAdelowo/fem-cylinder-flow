# DFG 2D-2: geometry, time-step and mesh convergence of the benchmark observables

Everything below is produced by `scripts/study_2d2_commands.sh` and tabulated with
`scripts/collect_2d2_study.py` from `results/2d2_<label>.json`. All runs use Taylor-Hood P2/P1,
Crank-Nicolson, Newton with the exact Jacobian (rtol = atol = 1e-10), and MUMPS LU. Observables
are taken from the last complete lift cycle (minimum of c_L to the next minimum), using cubic-spline
extrema. Every continued run below has a drift between its last two cycles of at most 7e-6 in
c_L,max and c_D,max (3e-5 for the 8 s runs from rest).

Canonical results: `results/2d2_<label>.txt` and `.json` for the labels `baseline_full_reanalysed`
(section 2), `geo_p1_full`, `geo_p2_full` (section 4), `pg_L2`, `pg_L3`, `pg_L4` (section 4),
`t_p2_dt01`, `t_p2_dt005`, `t_p2_dt0025` (which is also mesh level L3), `t_p2_dt00125` (section 5),
`sp_L1`, `sp_L2`, `sp_L4`, `sp_L5`, `sp_hc3`, `sp_hf2` (section 6) and `sp_L2_rest` (section 7). The JSON
files hold every per-cycle record, so all tables can be rebuilt from them with
`scripts/collect_2d2_study.py`; re-analysing a raw series with `scripts/analyze_2d2.py` reproduces
its JSON exactly (checked for `sp_L5`, `geo_p1_full` and `geo_p2_full`). The raw time series is kept only for the finest run (`sp_L5`, 297 CPU-minutes);
the others regenerate from `scripts/study_2d2_commands.sh`. The older files `results/2d2_unsteady_*`,
`results/2d2_timeseries_{smoke,local,full,full_dt01}.npz` and `figures/2d2_force_coefficients_*` are
from the original configuration and are kept unchanged as a historical record.

## 0. Where the reference values come from

Three kinds of numbers appear in this study and they must not be confused.

1. **Inherited from the project's earlier documentation.** The comparison intervals in
   `src/reference.py`: c_D,max [3.2200, 3.2400], c_L,max [0.9900, 1.0100], St [0.2950, 0.3050],
   p_diff [2.4600, 2.5000] (2D-2) and c_D [5.5700, 5.5900], c_L [0.0104, 0.0110],
   dP [0.1172, 0.1176] (2D-1). They have been the repository's intervals since its first commit and
   are attributed there to the "lower/upper bound" rows of Tables 3 and 4 of Schaefer and Turek
   (1996). **That paper was not re-read during this study**; these intervals were not
   independently re-verified, and nothing here shows them to be right or wrong.
2. **Taken from the benchmark maintainers' published material (FeatFlow, read 2026-09-21).**
   The definitions (force as the integral of nu grad(u) - p I over the circle, c = 2F/(Ubar^2 L),
   a cycle running from one minimum of c_L to the next, St = L f / Ubar, a start from the zero
   solution, measurement in a fully developed cycle between 25 s and 30 s); the 2D-1 high-order
   spectral values c_D = 5.57953523, c_L = 0.010618948, dP = 0.11752017 (Nabh 1998, as printed on
   the page); and the 2D-2 time series in `20141124_dfg_bench2_official.zip` ("new reference
   results", Q2/P1disc, Crank-Nicolson, higher-order line integration of the forces). The time series
   were analysed **here** with this repository's cycle extraction: level 6 (667,264 dofs),
   dt = 1/1600 gives c_D,max = 3.2274, c_L,max = 0.9866, period 0.33130 (St = 0.30184) and
   p_diff at c_L,max 2.4829. These are another code's numerical results, not certified bounds, and
   the maintainers' own finest values are not required to lie in the 1996 intervals.
3. **Recomputed by this repository.** Everything else, including every number in the tables below.

**Pressure difference.** The FeatFlow page defines p_diff(t) = p(0.15, 0.2) - p(0.25, 0.2) as a
time series over a cycle and gives no single instant. Whether the 1996 table specifies one (for
example half a period after the lift maximum) was **not verified**. This study therefore reports two
values, at the lift maximum and half a period later, and does not treat either as "the"
benchmark value.

**Measurement window.** The benchmark procedure prescribes about 25 s of simulation followed by
measurement between 25 s and 30 s. The runs here end at t = 12 s (t = 8 s for the runs from rest).
The periodicity evidence in section 7 supports the statement that the computed cycle is settled,
but a run that ends at 12 s is not the benchmark protocol evaluated at 25 to 30 s, and exact
protocol reproduction is not claimed.

## 1. Force definitions: what each is, and what is used where

Four force definitions are recorded for every run (`src/forces.py`). They are different
discretisations of one quantity.

| name | definition | relation to the benchmark and to the discrete problem |
|---|---|---|
| `laplacian` | F = -int (nu grad(u) - p I) n ds | the benchmark's stated traction, and the stress of the weak form solved here; closest to the benchmark's convention |
| `symmetric` | F = -int (nu (grad u + grad u^T) - p I) n ds | the repository's original choice; equals `laplacian` only where div u = 0, which Taylor-Hood satisfies only in the limit h -> 0 |
| `tangential` | the paper's tangential-derivative form | equals `laplacian` for div u = 0 on a no-slip wall |
| `variational` | residual of the discrete equations tested with a function equal to e_x (or e_y) on the cylinder nodes | most consistent with the discrete variational formulation: it is the exact force of the discrete system, uses no facet normals, and for Crank-Nicolson is centred at t_{n+1/2}; the traction forms are evaluated at t_{n+1} |

No definition is called correct here. `laplacian` was fixed as the default headline force in the
scripts before the unsteady results were known, because it is the benchmark's own definition;
`variational` is reported alongside it in every table because the steady 2D-1 comparison against
the spectral reference (section 8) shows it converging much faster, and the unsteady time-step
study (section 5) shows the traction forms carry a time-dependence that the variational force does
not. The two are shown side by side so that no result depends on which was chosen, and the spread
between definitions is read as a discretisation-error indicator. What matters is whether the
definitions approach one another under refinement; this holds for c_D,max (section 6) and does
**not yet** hold for c_L,max at the resolutions reached.

## 2. Starting point (original configuration)

Polygonal cylinder (degree-1 mesh), symmetric-stress force, dt = 0.005, mesh 12,020 cells /
54,860 dofs, run from rest to t = 8 s, maximum taken from raw time samples of the last half:
c_D,max = 3.2139, c_L,max = 0.9837, St = 0.3019. Re-analysed with the per-cycle extraction:
3.21389 / 0.98365 / 0.30172 (`results/2d2_baseline_full_reanalysed.txt`).

The original report called the last six lift peaks "flat to three significant figures" (spread
0.0012) as evidence of a converged limit cycle. That spread was mostly a sampling artefact: with
dt = 0.005 and a period of 0.331 s a raw sample can miss the crest by up to about 1e-3. With
spline-refined peaks the lift maxima still rise monotonically from cycle to cycle
(0.98291, 0.98327, ... 0.98367), so the flow was **near-periodic, not exactly periodic**, with a
remaining change of about 3e-5 per cycle at t = 8. The raw-sample maximum was low by 3e-4
(c_L,max) and 1e-4 (c_D,max) at this sampling.

## 3. Cylinder geometry

Measured through the finite-element geometry map that the solver uses
(`cylinder_geometry_report`). The imported degree is checked after `model_to_mesh`.

| h_cyl | facets | order 1: area error | order 1: max radial deviation | order 2: area error | order 2: max radial deviation |
|---:|---:|---:|---:|---:|---:|
| 0.015 | 21 | -1.49e-2 | 1.12e-2 R | -1.7e-5 | 1.6e-5 R |
| 0.005 | 63 | -1.66e-3 | 1.24e-3 R | -2.1e-7 | 1.9e-7 R |
| 0.002 | 158 | -2.6e-4 | 2.0e-4 R | -5.2e-9 | 4.9e-9 R |

At 63 facets the polygon's perimeter error is -4.1e-4. The order-1 mesh is an inscribed polygon
(vertices exactly on the circle, radial deviation equal to the chord sagitta R (1 - cos(pi/N)));
its enclosed area is short by (2 pi/N)^2 / 6 and the equivalent cylinder diameter by half that. The
order-2 mesh has the same vertices and cells, with edge nodes on the exact circle; its area error
falls at the h^4 rate in the table.

That is a statement about **representing the circle**. It says nothing about the order of the flow
solution: the velocity and pressure errors are governed by the P2/P1 discretisation and by the
resolved flow structure, and no fourth-order convergence of any flow quantity is claimed or
observed.

## 4. Geometry effect at identical vertices, cells, dofs and dt

Run from rest to t = 8 s, dt = 0.005, h_far = 0.03, h_cyl = 0.005 (12,020 cells, 54,860 dofs).

| force | polygon c_D,max | curved c_D,max | polygon c_L,max | curved c_L,max |
|---|---:|---:|---:|---:|
| laplacian | 3.20851 | 3.21758 | 0.98340 | 0.98587 |
| symmetric | 3.21389 | 3.22212 | 0.98365 | 0.98600 |
| tangential | 3.20312 | 3.21303 | 0.98314 | 0.98574 |
| variational | 3.22343 | 3.22672 | 0.98369 | 0.98539 |

At dt = 0.0025 and continued to t = 12, same vertices, polygon (`pg_*`) against curved (`sp_*`):

| level | dofs | polygon area error | variational c_D,max change | variational c_L,max change | Laplacian c_D,max / c_L,max change |
|---|---:|---:|---:|---:|---:|
| L2 | 24,741 | -3.7e-3 | +0.23 % | +0.40 % | +0.58 % / +0.56 % |
| L3 | 54,860 | -1.7e-3 | +0.10 % | +0.17 % | +0.28 % / +0.25 % |
| L4 | 121,989 | -7.3e-4 | +0.04 % | +0.08 % | +0.15 % / +0.13 % |

The effect of the curved boundary is about 1.0 to 1.1 times the polygon's area error for c_L,max
and about 0.6 times for c_D,max (variational force), at every level: it shrinks in proportion to
the geometric error and converges away under refinement. A polygon run at L5 was not made.

## 5. Time-step convergence

Curved geometry, base mesh (L3), all runs continued from the same developed t = 8 state to t = 12.

| dt | c_D,max lap. | c_D,max var. | c_L,max lap. | c_L,max var. | St |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 3.21772 | 3.22671 | 0.98564 | 0.98269 | 0.30082 |
| 0.005 | 3.21759 | 3.22673 | 0.98591 | 0.98543 | 0.30160 |
| 0.0025 | 3.21752 | 3.22673 | 0.98535 | 0.98610 | 0.30179 |
| 0.00125 | 3.21748 | 3.22673 | 0.98490 | 0.98627 | 0.30184 |
| observed order | - | - | not monotone | 2.01 | 2.00 |

* Second-order behaviour was observed for the **variational** c_L,max, for St and for the period.
  It is not generalised to the other observables or force definitions. Their Richardson
  estimates (from the last three levels; estimates, not computed values) are 0.30186 and 0.98633
  on this mesh. The variational c_L,max at dt = 0.01 (0.98269) is close to the maintainers' own
  dt = 1/100 value (0.98295).
* c_D,max depends only weakly on dt: it varies by 2.4e-4 (Laplacian) and 2e-5 (variational) across
  the four steps.
* The **traction-based c_L,max is not monotone in dt** (0.98564, 0.98591, 0.98535, 0.98490) and
  drifts the wrong way as dt decreases, whereas the variational force converges cleanly. A likely
  explanation, consistent with the data but not isolated by a separate experiment, is that the
  traction combines the viscous stress from u at t_{n+1} with the Crank-Nicolson pressure, which
  approximates t_{n+1/2}; the resulting O(dt) time-level mismatch would offset the O(dt^2) scheme
  error. The traction-based c_L,max changed by only 0.00027 between dt = 0.01 and 0.005 (the
  original repository's observation) while the converging force changes by 0.0027 over the same
  step, so the earlier conclusion that dt = 0.01 was already temporally converged was **not
  adequately supported**.

## 6. Mesh convergence

Curved geometry, dt = 0.0025, ratio 1.5 in both h_far and h_cyl, all continued from the developed
t = 8 state (interpolated onto each mesh) to t = 12. L3 is the run `t_p2_dt0025`. L5 is
`sp_L5`: 60,332 cells, 273,224 dofs, degree-2 geometry with 142 cylinder facets, dt = 0.0025,
1,600 steps from t = 8 to t = 12, initial state the L3 t = 8 state interpolated onto the L5 mesh,
2.0 Newton iterations per step, 17,837 s of CPU time (about 297 min; the timer does not include
periods when the machine slept, and elapsed wall time was inflated by other load, so CPU time is
the meaningful cost). It is not a checkpoint-resumed run.

| level | (h_far, h_cyl) | cells | dofs | c_D,max lap. | c_D,max var. | c_L,max lap. | c_L,max var. | St | p_diff @ c_L,max | p_diff @ +T/2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | (0.06, 0.01) | 3,077 | 14,234 | 3.19107 | 3.23609 | 0.98917 | 0.99715 | 0.30143 | 2.5017 | 2.5043 |
| L2 | (0.045, 0.0075) | 5,383 | 24,741 | 3.20138 | 3.22682 | 0.98461 | 0.98530 | 0.30174 | 2.4896 | 2.4924 |
| L3 | (0.03, 0.005) | 12,020 | 54,860 | 3.21752 | 3.22673 | 0.98535 | 0.98610 | 0.30179 | 2.4732 | 2.4753 |
| L4 | (0.02, 0.00333) | 26,852 | 121,989 | 3.22327 | 3.22746 | 0.98709 | 0.98644 | 0.30178 | 2.4780 | 2.4806 |
| L5 | (0.0133, 0.00222) | 60,332 | 273,224 | 3.22584 | 3.22761 | 0.98760 | 0.98650 | 0.30177 | 2.4833 | 2.4856 |

The other two traction definitions at L5: c_D,max 3.22613 (symmetric), 3.22555 (tangential);
c_L,max 0.98785 (symmetric), 0.98736 (tangential). The four definitions therefore span
3.22555 to 3.22761 (0.06 %) in c_D,max and 0.98650 to 0.98785 (0.14 %) in c_L,max at L5.

Observed order and Richardson estimate for **every** consecutive triple of levels (ratio 1.5; an
estimate is printed only if the sequence is monotone and 0.5 <= p <= 4):

| quantity | L1-L3 | L2-L4 | L3-L5 |
|---|---|---|---|
| c_D,max variational | p = 11.5, outside range | non-monotone | p = 4.05, outside range |
| c_L,max variational | non-monotone | p = 2.09, estimate 0.98670 | p = 4.59, outside range |
| c_D,max Laplacian | p = -1.1 | p = 2.54, estimate 3.22646 | p = 1.99, estimate 3.22790 |
| c_L,max Laplacian | non-monotone | p = -2.1 | p = 2.99, estimate 0.98782 |
| c_D,max symmetric / tangential | - | 3.03 / 2.38 (3.22571 / 3.22739) | 1.37 / 2.21 (3.22813 / 3.22806) |
| c_L,max symmetric / tangential | - | -3.15 / -1.20 | 3.85 / 2.19 (0.98796 / 0.98780) |
| St | p = 4.6, outside range | non-monotone | p = 3.14, estimate 0.30177 |

What L5 changes and what it does not:

* **Variational c_L,max** is monotone from L2 on (0.98530, 0.98610, 0.98644, 0.98650), and the
  L4 to L5 change is only 6e-5 (0.006 %), so L4 is already effectively converged for this force at
  this dt. But the **observed order is not stable** (2.09 from L2 to L4, 4.59 from L3 to L5), so no
  asymptotic rate is claimed. The earlier estimate of 0.9867 from L2 to L4 overshot the value
  actually computed at L5 (0.98650) and is withdrawn as a limit.
* **Variational c_D,max** is not monotone (3.23609, 3.22682, 3.22673, 3.22746, 3.22761); the change
  from L4 to L5 is 1.4e-4 (0.005 %). It is effectively converged at the 2e-4 level with no usable
  order.
* **The traction forms have not settled.** Their c_L,max is still rising (Laplacian 0.98461,
  0.98535, 0.98709, 0.98760) and their three-level extrapolations (0.98780 to 0.98796) lie about
  0.13 % above the variational value 0.98650. At these resolutions and dt = 0.0025 the definitions
  therefore do **not** agree on c_L,max to better than about 0.14 %. The traction c_L,max also has
  the non-monotone time-step behaviour described in section 5, which may account for part of the
  difference (the traction value decreased by 4.5e-4 on the last dt halving on the L3 mesh); this was
  not separated by a further experiment at L5.
* **c_D,max** does show the definitions approaching one another: the three traction extrapolations
  (3.2279 to 3.2281) lie within 0.02 % of the variational 3.2276.
* **St** changes by 2e-5 over L3 to L5 and by 1e-5 from L4 to L5 (0.30179, 0.30178, 0.30177) at
  dt = 0.0025.
* **p_diff** is not converging monotonically (2.4732, 2.4780, 2.4833 over L3 to L5; the L4 to L5
  change is 5e-3, 0.2 %), so no order is given. It is the P1 pressure evaluated at two points on the
  cylinder surface and its non-monotone behaviour was not investigated further.
* Coarse-mesh values landing inside an interval mean little: the variational c_L,max is 0.99715
  (inside [0.99, 1.01]) on L1 and 0.9853 on L2.

Decoupling the two mesh sizes (curved, dt = 0.0025):

| mesh | (h_far, h_cyl) | dofs | c_D,max lap. | c_L,max lap. | c_D,max var. | c_L,max var. |
|---|---|---:|---:|---:|---:|---:|
| L3 | (0.03, 0.005) | 54,860 | 3.21752 | 0.98535 | 3.22673 | 0.98610 |
| finer at cylinder | (0.03, 0.00333) | 99,167 | 3.22306 | 0.98701 | 3.22751 | 0.98667 |
| finer wake | (0.02, 0.005) | 71,888 | 3.21429 | 0.98413 | 3.22651 | 0.98507 |
| both | (0.02, 0.00333) | 121,989 | 3.22327 | 0.98709 | 3.22746 | 0.98644 |

Refining only the cylinder region raises the Laplacian c_D,max and c_L,max by 0.17 percent each;
refining only the wake lowers them by 0.10 and 0.12 percent. For the variational force c_D,max
changes by at most 0.02 percent in either case and c_L,max by +0.06 percent (cylinder region) and
-0.10 percent (wake). Both mesh sizes matter for c_L,max at the 0.1 percent level, and wake
refinement does not move it in the same direction as cylinder refinement.

## 7. Periodic regime and Strouhal number

* Every run reports the change between its last two lift cycles: at most 7e-6 in c_L,max and
  c_D,max in the continued runs (L5: 1e-7), and 3e-5 in the 8 s runs from rest. A 32-cycle
  from-rest run on L2 has drift 3e-8.
* The interpolated-restart procedure was validated on L2: restart from the t = 8 base-mesh state
  against a run from rest to t = 12 agree to 1.5e-6 in c_L,max, 6e-7 in c_D,max and 3.5e-8 in the
  period.
* St is computed three ways (mean cycle period, zero-padded Hann-windowed FFT with quadratic peak
  interpolation over a 3 to 4 s window, harmonic least-squares fit); the three agree to 1e-5 in
  every run. Definition: St = f D / Ubar with f the reciprocal of the mean period over the last
  four cycles.
* The largest raw time sample of c_L underestimates the cycle maximum by 3e-4 at dt = 0.005 and by
  4e-5 at dt = 0.00125; the spline extremum differs from an independent local-parabola fit by 1e-4
  for the committed dt = 0.01 series and 2e-7 at dt = 0.00125.

These results support the statement that the computed cycle is settled. They do not make a run
ending at t = 12 s identical to a benchmark evaluated at 25 to 30 s (section 0).

## 8. Steady 2D-1 recheck

`scripts/run_2d1_steady.py --config full`, four-level ladder up to 82,913 cells / 375,281 dofs.
Spectral reference (Nabh 1998, FeatFlow page): c_D = 5.57953523, c_L = 0.010618948, dP = 0.11752017.

| configuration (finest mesh) | c_D | c_L | dP |
|---|---:|---:|---:|
| original (polygon, symmetric) | 5.57435 | 0.010427 | 0.11750 |
| polygon, laplacian | 5.57288 | 0.010543 | 0.11750 |
| curved, laplacian | 5.57829 | 0.010489 | 0.11751 |
| curved, variational | 5.57953 | 0.010619 | 0.11751 |

Curved variational relative to the spectral values: c_D -3.6e-7, c_L +1.8e-5, dP -1.0e-4 (dP is
force-independent; the curved and polygon meshes give -1.0e-4 and -2.1e-4). All three quantities are
inside the intervals in every configuration. The curved variational c_D is monotone over the ladder
(5.579089, 5.579458, 5.579525, 5.579533) but c_L is not (0.010722, 0.010606, 0.010621, 0.010619).
The traction-based c_L converges slowly (polygon symmetric: 0.01275, 0.01063, 0.01119, 0.01043, not
monotone; curved laplacian 0.01255, 0.01170, 0.01107, 0.01049 with the finest value 1.2 % below the
spectral one), so the original agreement of c_L with its interval at the finest level was correct
but not a clean convergence. Only about five significant figures of c_L are meaningful.

## 9. Manufactured solutions

`results/mms_verification_post_changes.txt`: Check B unchanged to every printed digit (velocity L2
2.998, H1 1.997, pressure L2 2.131 at N = 64); Check A at rounding error (1e-13 to 1e-16).

## 10. Assessment of each 2D-2 observable

Values are the finest computed (L5, curved geometry, dt = 0.0025, last lift cycle); "FeatFlow" is
the level-6, dt = 1/1600 entry re-analysed here (dt = 0.0025 leaves a temporal error, section 5).
Intervals are the inherited ones of section 0.

| | c_D,max | c_L,max | St | p_diff |
|---|---|---|---|---|
| finest computed | 3.22761 var., 3.22584 lap. (3.22555 to 3.22761 over the four definitions) | 0.98650 var., 0.98760 lap. (0.98650 to 0.98785) | 0.30177 | 2.4833 at c_L,max, 2.4856 at +T/2 |
| convergence | var.: L4 to L5 change 1.4e-4, not monotone, no order; traction forms monotone, order about 2, extrapolations 3.2279 to 3.2281 | var.: monotone, L4 to L5 change 6e-5, order not stable (2.1, 4.6); traction forms still rising, extrapolations 0.9878 to 0.9880 | changes by 2e-5 over L3 to L5 | not monotone; L4 to L5 change 5e-3; not shown to converge |
| inherited interval | [3.22, 3.24]: inside for all four definitions | [0.99, 1.01]: below by 0.35 % (var.) to 0.22 % (0.98785, the highest definition) | [0.295, 0.305]: inside | [2.46, 2.50]: inside at L2 to L5 for both instants (L1 is outside) |
| FeatFlow | 3.22739: +0.007 % (var.), -0.05 % (lap.) | 0.98658: -0.008 % (var.), +0.10 % (lap.) | 0.30184: -0.02 % | 2.4829: +0.016 % |
| remaining uncertainty | force-definition spread 0.06 %; no mesh order | force-definition spread 0.14 %; temporal effect about +2e-4 on the variational value (section 5); the traction time-dependence unresolved | temporal effect about +7e-5 | non-monotone in h, definition of the instant unverified, no order |

Indicative only, assuming the spatial and temporal errors add: combining the L5 variational values
with the base-mesh time-step estimates of section 5 gives c_L,max about 0.9867, c_D,max about
3.2276 and St about 0.30184. These are estimates, not computed results, and the additivity is an
assumption that was not tested.

Statement by quantity:

* **c_D,max.** The original 3.2139 was low for numerical reasons: the polygonal cylinder
  (+0.28 % when replaced by the curved one at that resolution), the symmetric-stress traction on a
  coarse mesh, and raw-sample extraction (1e-4). The refined values, 3.2256 to 3.2276 depending on
  definition, lie inside the inherited interval and within 0.06 % of the FeatFlow computation.
* **c_L,max.** The refined value remains below the inherited 0.99 lower bound, at 0.9865 to 0.9879
  depending on the force definition, and the variational value agrees closely with the FeatFlow
  computation examined here (0.9866). The force definitions differ by 0.14 % at this resolution
  and that difference is unresolved. Because the interval's source was not re-read and the
  benchmark protocol (time window) was not exactly followed, this study neither identifies a
  defect in the implementation nor a problem with the interval.
* **St.** Changes by 2e-5 over the last three meshes, second order in dt, inside the inherited
  interval and within 0.02 % of the FeatFlow value, three independent estimators agreeing.
* **p_diff.** Inside the inherited interval for both instants at L2 to L5, close to the FeatFlow
  value at L5, but the mesh sequence is not monotone and the intended time instant is not
  verified. It is reported as within the interval, not as converged.

## 11. Limitations

* The 1996 intervals and the p_diff time definition are unverified against the primary source.
* Runs end at t = 12 s (8 s from rest), not at the benchmark's 25 to 30 s.
* The mesh study has five levels at one time step; time step and mesh were separated but not
  combined into a joint extrapolation. Orders are not stable across triples for the variational
  c_L,max and c_D,max.
* The traction forms' c_L,max and time-step behaviour are not understood beyond the hypothesis in
  section 5, and at L5 they differ from the variational value by 0.14 %.
* There is no polygon run at L5, no time-step study at L4 or L5, and the mesh family has one
  refinement ratio.
* Wall-clock times were inflated by other processes on the machine and by machine sleep; CPU times
  are recorded in each `results/2d2_<label>.json`.
* Only serial execution was run.
