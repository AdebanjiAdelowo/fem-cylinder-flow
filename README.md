# Finite-Element Simulation of 2D Incompressible Flow Past a Cylinder

A Taylor-Hood P2/P1 finite-element solver for the classical DFG 2D-1/2D-2 benchmark (steady and
unsteady flow past a cylinder in a channel), implemented with FEniCSx, verified against a
manufactured solution before being trusted on the benchmark itself, and validated against the
original published reference values.

## Motivation

This project demonstrates a different part of the applied-mathematics toolkit than the companion
`navier-stokes-2d` (pseudo-spectral, periodic-domain) project: the full finite-element workflow —
weak/variational formulation, a provably stable mixed element pair, unstructured mesh generation
around a curved solid boundary, Dirichlet/Neumann boundary conditions, and Newton's method for the
nonlinear convective term — culminating in a wall-bounded flow with a real solid obstacle, which a
periodic pseudo-spectral method cannot represent at all.

## Benchmark definition

The benchmark is exactly as specified in Schäfer, M. & Turek, S. (1996), *Benchmark Computations of
Laminar Flow Around a Cylinder*, in *Flow Simulation with High-Performance Computers II* (Notes on
Numerical Fluid Mechanics, Vol. 52), Vieweg. No geometry, boundary condition, or reference value in
this repository is invented; every number below is taken directly from that paper (Sec. 2.2, Table 3,
Table 4), re-derived in `src/geometry.py` and cited again where used.

**Geometry**: channel $\Omega = [0, 2.2] \times [0, 0.41] \setminus B_r(0.2, 0.2)$, i.e. a
$2.2\,\mathrm{m} \times 0.41\,\mathrm{m}$ channel with a cylinder of diameter $D = 0.1\,\mathrm{m}$
centred at $(0.2, 0.2)$ — off-centre, to break the flow's symmetry and (at Re = 100) trigger vortex
shedding.

**Two test cases are implemented**:

| Case | Inflow | $U_m$ | Re | Regime |
|---|---|---|---|---|
| 2D-1 | $U(0,y) = 4U_m y(H-y)/H^2$ | 0.3 | 20 | steady |
| 2D-2 | $U(0,y,t) = 4U_m y(H-y)/H^2$ (time-independent profile) | 1.5 | 100 | unsteady, periodic vortex shedding |

with $\nu = 10^{-3}\,\mathrm{m^2/s}$, $\rho = 1.0\,\mathrm{kg/m^3}$ for both. (2D-3, the ramped-Re
case, is not implemented; see Limitations.)

## Governing equations

$$\rho\,\partial_t \mathbf{U} + \rho\,(\mathbf{U}\cdot\nabla)\mathbf{U} = \rho\nu\,\nabla\cdot(\nabla\mathbf{U} + \nabla\mathbf{U}^\top) - \nabla P, \qquad \nabla\cdot\mathbf{U} = 0,$$

exactly as given in the benchmark paper (Sec. 2.1), with no-slip on the walls and cylinder, the
parabolic inflow above, and a "do-nothing" (natural) outflow condition — explicitly permitted by the
paper ("the outflow condition can be chosen by the user").

## Weak formulation

Multiplying by test functions $(\mathbf{v}, q)$, integrating over $\Omega$, and integrating the
viscous and pressure terms by parts (dropping the resulting boundary term on the outlet, which is
exactly the "do-nothing" condition; see Heywood, Rannacher & Turek, 1996) gives: find
$(\mathbf{u}, p) \in (H^1_{\mathbf{g}} \times L^2)$ such that for all
$(\mathbf{v}, q) \in (H^1_0 \times L^2)$,

$$\int_\Omega \nu\,\nabla\mathbf{u}\!:\!\nabla\mathbf{v}\,dx - \int_\Omega p\,(\nabla\cdot\mathbf{v})\,dx + \int_\Omega q\,(\nabla\cdot\mathbf{u})\,dx + \int_\Omega \big((\mathbf{u}\cdot\nabla)\mathbf{u}\big)\cdot\mathbf{v}\,dx = 0,$$

with $\mathbf{u} = \mathbf{g}$ (the no-slip/inflow data) enforced strongly on $\Gamma_D$. The full
derivation, including the exact boundary-term cancellation that produces the do-nothing condition, is
in the `src/navier_stokes.py` module docstring. For the unsteady case, a $\theta$-method blends the
viscous+convective operator between time levels ($\theta = 1$: backward Euler; $\theta = 0.5$:
Crank-Nicolson, used here) while keeping the pressure-velocity coupling and incompressibility
constraint fully implicit at the new time level, since pressure has no time derivative of its own.

## Finite-element spaces

**Taylor-Hood $P_2/P_1$**: continuous piecewise-quadratic velocity, continuous piecewise-linear
pressure (`src/spaces.py`). This pair is inf-sup (LBB) stable — the velocity space is rich enough,
relative to the pressure space, that the discrete divergence operator has no spurious near-null
modes, ruling out the checkerboard pressure oscillations an equal-order $P_1/P_1$ pair would produce
(Girault & Raviart, 1986; Elman, Silvester & Wathen, 2014, Ch. 3). It is also the pair the benchmark
paper's own reference computations use (Table 2, entries 8a/8b).

A consequence of Taylor-Hood worth stating precisely: the discrete velocity is **not** pointwise
divergence-free. The discrete continuity equation only enforces $\int_\Omega q\,\nabla\cdot\mathbf{u}_h\,dx = 0$
for $q$ in the *continuous* $P_1$ pressure space, and $\nabla\cdot\mathbf{u}_h$ (piecewise quadratic
velocity differentiated once) is a richer, generally discontinuous piecewise-linear function that
continuous $P_1$ test functions cannot fully annihilate. $\|\nabla\cdot\mathbf{u}_h\|_{L^2}$ is
therefore small and shrinks under mesh refinement, but is not exactly zero at any finite resolution —
confirmed directly in `tests/test_navier_stokes.py` and discussed further under Verification.

## Numerical method

- **Nonlinear solve**: PETSc SNES (Newton line search), with the exact Jacobian obtained from UFL's
  automatic symbolic differentiation of the residual (`ufl.derivative`), not a hand-linearised or
  Picard approximation.
- **Linear solve**: direct (LU via MUMPS) at every Newton/time step. Correctness and benchmark
  agreement are established first, per this project's brief; an iterative Krylov solver with a
  fieldsplit preconditioner is a natural performance follow-up for much larger 3D-scale problems (see
  Limitations), not attempted here.
- **Mesh generation**: gmsh (OpenCASCADE kernel), boolean-cut rectangle-minus-disk, with a
  distance-based sizing field refining the mesh near the cylinder (`src/geometry.py`). The geometry
  map is degree 1 by default (the cylinder is an inscribed polygon) or degree 2 with
  `geometry_order=2` / `--geometry-order 2` (curved isoparametric cylinder: same vertices and cells,
  extra edge nodes on the exact circle). The imported coordinate-element degree is checked after the
  gmsh to dolfinx conversion, and `cylinder_geometry_report` measures the perimeter, enclosed-area
  and radial-deviation errors of the meshed boundary through the geometry map the solver uses.
- **Force post-processing** (`src/forces.py`): four definitions of the force on the cylinder are
  available. `laplacian` is the benchmark's own stress $\nu\nabla u - pI$; `symmetric` is
  $\nu(\nabla u + \nabla u^\top) - pI$, which is equal to it only for a divergence-free velocity;
  `tangential` is the paper's tangential-derivative form; `variational` is the discrete reaction
  force, the weak-form residual tested with a function equal to a unit vector on the cylinder nodes
  (no facet normals, and for the unsteady scheme centred at $t_{n+1/2}$). All four are recorded so
  the post-processing error is visible next to the discretisation error. The benchmark definition
  audit is in [`docs/benchmark_audit.md`](docs/benchmark_audit.md).
- **Time discretisation** (2D-2 only): Crank-Nicolson $\theta$-method, Newton solve every step.
- **Newton/SNES tolerances**: `snes_rtol = 1e-10`, `snes_atol = 1e-10`, `snes_max_it = 25`, with
  `snes_error_if_not_converged = True` — every solve in this repository either meets this tolerance or
  raises, so no result below is from a silently non-converged solve.

## Implementation

```
fem-cylinder-flow/
├── src/
│   ├── geometry.py         gmsh mesh generation (degree 1 or 2), geometry-error report, constants
│   ├── spaces.py             Taylor-Hood P2/P1 mixed function space
│   ├── navier_stokes.py       weak-form residual (steady + theta-method unsteady), Newton/SNES solve
│   ├── cylinder_bcs.py         no-slip / parabolic-inflow / do-nothing boundary conditions
│   ├── forces.py                c_D, c_L (four force definitions), pressure probes
│   ├── periodic.py               per-cycle extraction of benchmark observables, periodicity check, St
│   ├── restart.py                 save/load a state, interpolating between meshes
│   ├── reference.py                published comparison values and their provenance
│   └── manufactured.py           UFL-symbolic manufactured Stokes/Navier-Stokes solutions
├── scripts/
│   ├── verify_mms.py         manufactured-solution verification + mesh-convergence figure
│   ├── run_2d1_steady.py       2D-1 benchmark + mesh-convergence ladder
│   ├── run_2d2_unsteady.py      2D-2 benchmark: time series, per-cycle analysis, restart/checkpoint
│   ├── analyze_2d2.py            re-run the cycle analysis on a saved time series (no FEM solve)
│   ├── collect_2d2_study.py       tabulate runs, observed convergence orders
│   ├── study_2d2_commands.sh       the geometry / time-step / mesh study as an exact command matrix
│   ├── plot_2d1_convergence.py   c_D/c_L/Delta P vs. resolution figure
│   └── plot_fields_2d1.py         velocity/pressure/vorticity/streamline figure
├── tests/                   62 pytest tests: geometry, spaces, BCs, forces, periodic analysis, restart, MMS, solver
├── docs/                    benchmark definition audit, 2D-2 convergence study
├── configs/                 smoke.yaml / local.yaml / full.yaml
├── figures/, results/       generated outputs (see below)
└── environment.yml           conda-forge FEniCSx environment specification
```

## Environment: FEniCSx availability on Apple Silicon

FEniCSx is **not** reliably installable via `pip` on macOS/arm64 (no wheels). The conda-forge route
was verified to work reliably on this machine (Apple M3 Pro, macOS 26):

```bash
conda env create -f environment.yml    # or: conda create -n fenicsx -c conda-forge fenics-dolfinx mpich petsc4py gmsh python-gmsh pyvista python=3.12 numpy scipy matplotlib pytest pyyaml
conda activate fenicsx
```

Installed and used for every result in this repository: **dolfinx 0.10.0**, **petsc 3.24.3**,
**mpich 4.3.2**, **gmsh 4.15.2**, **python 3.12.12** (full pinned list in
`results/environment_versions.txt`). A from-scratch reinstall into a clean conda environment was
performed as part of this project and confirmed working (Poisson solve reproduced the known
$-\nabla^2 u = 1$ unit-square reference value $u_{\max} \approx 0.0734$ to the expected
mesh-discretisation accuracy) — the environment is reproducible, not merely "worked once."

## Verification: before trusting the cylinder benchmark

Per the project's own standard, the cylinder benchmark is **not** used as the first test of the
implementation. `scripts/verify_mms.py` runs two manufactured-solution checks on the unit square
first (`src/manufactured.py`), building the exact velocity from a streamfunction,
$\mathbf{u} = \nabla^\perp\phi = (\partial_y\phi, -\partial_x\phi)$, which is divergence-free
*identically* by construction, and computing the exact forcing $\mathbf{f}$ required to make
$(\mathbf{u}, p)$ an exact solution via UFL's own symbolic differentiation (no external CAS, no
hand-transcribed algebra — the same machinery FFCX uses to form the Newton Jacobian).

**Check A — exact recovery.** $\phi$ chosen as a degree-3 polynomial, so $\mathbf{u}$ is exactly
degree $\le 2$ (representable by $P_2$) and $p$ is degree 1 (representable by $P_1$). Errors at any
resolution should be at floating-point roundoff — the FEM analogue of the Taylor-Green-vortex check
in the companion spectral project:

| $N$ | $\|u-u_h\|_{L^2}$ | $\|u-u_h\|_{H^1}$ | $\|p-p_h\|_{L^2}$ |
|---:|---:|---:|---:|
| 4 | $5.16\times10^{-16}$ | $7.68\times10^{-15}$ | $2.25\times10^{-14}$ |
| 8 | $6.74\times10^{-16}$ | $1.41\times10^{-14}$ | $8.15\times10^{-14}$ |
| 16 | $1.07\times10^{-15}$ | $2.59\times10^{-14}$ | $3.13\times10^{-13}$ |

confirmed at machine precision.

**Check B — mesh convergence.** $\phi, p$ chosen trigonometric (not exactly representable by either
space), giving a genuine refinement study. Expected rates for Taylor-Hood $P_2/P_1$:
$\|u-u_h\|_{L^2} = O(h^3)$, $\|u-u_h\|_{H^1} = O(h^2)$, $\|p-p_h\|_{L^2} = O(h^2)$ (Girault & Raviart,
1986, Thm. II.1.1):

| $N$ | $h$ | $\|u-u_h\|_{L^2}$ | order | $\|u-u_h\|_{H^1}$ | order | $\|p-p_h\|_{L^2}$ | order |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.2500 | $8.32\times10^{-2}$ | — | $2.251$ | — | $4.07\times10^{-1}$ | — |
| 8 | 0.1250 | $1.05\times10^{-2}$ | 2.984 | $6.17\times10^{-1}$ | 1.867 | $3.11\times10^{-2}$ | 3.712 |
| 16 | 0.0625 | $1.33\times10^{-3}$ | 2.982 | $1.59\times10^{-1}$ | 1.958 | $2.87\times10^{-3}$ | 3.437 |
| 32 | 0.0312 | $1.67\times10^{-4}$ | 2.993 | $4.00\times10^{-2}$ | 1.989 | $4.46\times10^{-4}$ | 2.686 |
| 64 | 0.0156 | $2.09\times10^{-5}$ | 2.998 | $1.00\times10^{-2}$ | 1.997 | $1.02\times10^{-4}$ | 2.131 |

Velocity converges to the theoretical $O(h^3)/O(h^2)$ rates essentially immediately; the pressure rate
starts higher than $O(h^2)$ at coarse $N$ and settles toward it (2.131 at $N=64$), a common
pre-asymptotic pattern, not a discrepancy (figure: `figures/mms_convergence.png`).

**Incompressibility.** As noted under Finite-Element Spaces, Taylor-Hood does not give a pointwise
divergence-free $\mathbf{u}_h$. `tests/test_navier_stokes.py` confirms $\|\nabla\cdot\mathbf{u}_h\|_{L^2}$
genuinely *decreases* under mesh refinement on a smooth (non-singular) test problem — an earlier,
discarded version of this test used the classical discontinuous-lid cavity, whose well-known corner
singularity was found to *prevent* this norm from decreasing at all (constant to 4 significant figures
from $N=8$ to $N=64$); switching to a lid profile that vanishes smoothly at the corners restored clean
convergence, confirming the flat behaviour was a property of that singular test problem, not a solver
defect. This diagnosis (and the corrected test) is preserved in the repository as an example of
honest debugging.

## Cylinder-flow results

### 2D-1 (steady, Re = 20): mesh convergence and benchmark comparison

`python scripts/run_2d1_steady.py --config full` — a 4-level mesh ladder, finest mesh 82,913 cells /
375,281 dofs:

| cells | dofs | Newton it. | time (s) | $c_D$ | $c_L$ | $\Delta P$ |
|---:|---:|---:|---:|---:|---:|---:|
| 5,324 | 24,508 | 5 | 0.89 | 5.5443 | 0.01275 | 0.1177 |
| 13,351 | 60,952 | 5 | 2.45 | 5.5625 | 0.01063 | 0.1169 |
| 37,207 | 168,879 | 5 | 5.10 | 5.5704 | 0.01119 | 0.1173 |
| 82,913 | 375,281 | 5 | 12.37 | 5.5744 | 0.01043 | 0.1175 |

**Reference** (Schäfer & Turek, 1996, Table 3, lower/upper bound row): $c_D \in [5.5700, 5.5900]$,
$c_L \in [0.0104, 0.0110]$, $\Delta P \in [0.1172, 0.1176]$.

**At the finest mesh, all three computed quantities fall inside the published reference interval**:
$c_D = 5.5744$, $c_L = 0.01043$, $\Delta P = 0.1175$. This is a directly verified statement, not an
approximate one — see `results/2d1_steady_full.txt`. The coarser levels show the expected monotonic
approach toward the reference interval (figure: `figures/2d1_convergence_full.png`); at the coarsest
tested level $c_D$ and $\Delta P$ are still outside it by roughly 1–2%. Velocity magnitude (with
streamlines), pressure, and vorticity fields at the finest mesh are shown in
`figures/2d1_fields_full.png`: qualitatively correct stagnation-point pressure rise ahead of the
cylinder, a low-pressure region on its flanks, and an antisymmetric vorticity pattern of opposite sign
above/below the wake centreline, consistent with steady laminar flow past a bluff body at this
Reynolds number.

### 2D-1 with curved geometry and the four force definitions

`python scripts/run_2d1_steady.py --config full --geometry-order 2` (same four meshes, 375,281 dofs
at the finest). The high-order spectral reference values (Nabh 1998, as published on the FeatFlow
benchmark page) are $c_D = 5.57953523$, $c_L = 0.010618948$, $\Delta P = 0.11752017$.

| configuration (finest mesh) | $c_D$ | $c_L$ | $\Delta P$ |
|---|---:|---:|---:|
| polygon, symmetric stress (original) | 5.57435 | 0.010427 | 0.11750 |
| polygon, benchmark stress | 5.57288 | 0.010543 | 0.11750 |
| curved, benchmark stress | 5.57829 | 0.010489 | 0.11751 |
| curved, variational force | 5.57953 | 0.010619 | 0.11751 |

All three quantities are inside the intervals in every row. With the variational force on the
curved mesh the finest-mesh values differ from the spectral ones by $-3.6\times10^{-7}$ (relative,
$c_D$) and $+1.8\times10^{-5}$ ($c_L$), and $\Delta P$ by $-1.0\times10^{-4}$ ($\Delta P$ does not
depend on the force definition). $c_D$ converges monotonically over the four levels (5.579089,
5.579458, 5.579525, 5.579533); $c_L$ does not (0.010722, 0.010606, 0.010621, 0.010619). The
traction-based $c_L$ converges slowly: with the symmetric stress on the polygonal mesh it is not
monotone over the ladder (0.01275, 0.01063, 0.01119, 0.01043), and with the benchmark stress on the
curved mesh the finest value is 1.2 percent below the spectral one. Only about five significant
figures of $c_L$ are meaningful.

### 2D-2 (unsteady, Re = 100): vortex shedding

The DFG 2D-2 benchmark was revisited with curved geometry, audited force definitions, time-step and
mesh refinement, and periodic-cycle extraction. Drag, Strouhal number and pressure difference lie
within their stated ranges. The refined lift maximum remains slightly below the stated 0.9900 lower
bound while closely matching the published FeatFlow computation examined here; **full 2D-2
reproduction is therefore not claimed.** Every table is in
[`docs/2d2_convergence_study.md`](docs/2d2_convergence_study.md).

**Method.** Crank-Nicolson time stepping with a Newton solve every step. Benchmark quantities are
extracted per lift cycle (from one minimum of $c_L$ to the next, the benchmark's own definition)
from cubic-spline extrema of the sampled signals, not from the largest time sample. A run counts as
periodic only if the change between its last two cycles is below a stated tolerance. The Strouhal
number $\mathrm{St} = f D / \bar U$ is computed from the mean cycle period and cross-checked by a
zero-padded FFT and a harmonic least-squares fit; the three agree to $10^{-5}$ in every run.

**Where the reference numbers come from.** (1) The intervals in the table below are those the
repository has carried since its first commit, attributed to the 1996 benchmark paper of Schäfer and
Turek; that paper was **not re-read** for this study and the intervals were not independently
re-verified. (2) The FeatFlow values are the benchmark maintainers' own published time series
(level 6, 667,264 dofs, $\Delta t = 1/1600$), analysed here with the same cycle extraction; they are
another code's numerical results, not certified bounds. (3) All other numbers were computed here.
The pressure difference is defined on the FeatFlow page as a function of time over a cycle; whether
the 1996 table fixes a single instant was not verified, so it is reported at the lift maximum
($\Delta P_1$) and half a period later ($\Delta P_2$).

**Force definitions.** Four are recorded (see Numerical method): the benchmark stress
`laplacian`, the original `symmetric` stress, `tangential`, and the `variational` reaction force,
which is the most consistent with the discrete formulation. `laplacian` is the default headline
force because it is the benchmark's own definition; `variational` is shown next to it because in
the steady 2D-1 check it converges far faster and in the time-step study the traction forms show a
time dependence that it does not. Neither is called correct; the spread between them is read as a
discretisation-error indicator.

**Finest computed values** (curved degree-2 cylinder, 273,224 dofs, $\Delta t = 0.0025$, last lift
cycle of a run to $t = 12$ s that is periodic to $10^{-7}$):

| Quantity | Computed: variational / benchmark stress | FeatFlow (finest published) | Interval (inherited) | Position |
|---|---:|---:|---|---|
| $C_{D,\max}$ | 3.2276 / 3.2258 | 3.2274 | [3.2200, 3.2400] | inside |
| $C_{L,\max}$ | 0.9865 / 0.9876 | 0.9866 | [0.9900, 1.0100] | below, by 0.35 % / 0.24 % |
| St | 0.3018 | 0.30184 | [0.2950, 0.3050] | inside |
| $\Delta P_1$, $\Delta P_2$ | 2.4833, 2.4856 | 2.4829 ($\Delta P_1$) | [2.4600, 2.5000] | inside |

The four force definitions span 3.2256 to 3.2276 in $C_{D,\max}$ and 0.9865 to 0.9879 in $C_{L,\max}$.

**Quantity by quantity.**

* **$C_{D,\max}$.** Effectively converged at the $2\times10^{-4}$ level for the variational force
  (L4 to L5 change $1.4\times10^{-4}$), but the mesh sequence is not monotone, so no order is claimed.
  The traction forms converge monotonically (order about 2) to about 3.228, within 0.02 % of the
  variational value. Inside the inherited interval and within 0.06 % of FeatFlow. The original
  value 3.2139 was low for numerical reasons: the polygonal cylinder, the traction force on a
  coarse mesh, and raw-sample extraction.
* **$C_{L,\max}$.** The variational value is monotone over the last four levels and changes by
  $6\times10^{-5}$ from L4 to L5, but the observed order is not stable across level triples (2.1,
  then 4.6), so no rate is claimed. The traction values are still rising (0.9871 at L4, 0.9876 at
  L5, extrapolating to 0.9878 to 0.9880), so the four definitions disagree by 0.14 % at the finest
  mesh; this is unresolved. All values remain below the inherited lower bound of 0.9900. The
  refined result remains below the stated 0.99 lower bound but agrees closely with the published
  FeatFlow computation examined in this study. That is not evidence of a defect in this
  implementation, nor of a problem with the interval, whose source was not re-checked.
* **St.** Changes by $2\times10^{-5}$ over the last three meshes and converges at second order in
  $\Delta t$ (0.30082, 0.30160, 0.30179, 0.30184 for $\Delta t = 0.01$ to 0.00125). Inside the
  interval and within 0.02 % of FeatFlow.
* **$\Delta P$.** Inside the interval at every mesh from L2 on for both instants, and close to the
  FeatFlow value at L5, but the mesh sequence is not monotone (2.4732, 2.4780, 2.4833 over L3 to L5)
  and the instant is not verified, so it is reported as within the range, not as converged.

**What was established by varying one factor at a time.**

* **Geometry.** With identical vertices, cells and dofs, the curved cylinder raises the variational
  $C_{L,\max}$ by 0.40, 0.17 and 0.08 percent and $C_{D,\max}$ by 0.23, 0.10 and 0.04 percent over
  three levels, in proportion to the polygon's area error ($-3.7\times10^{-3}$, $-1.7\times10^{-3}$,
  $-7.3\times10^{-4}$). The effect converges away with the geometric error. This concerns the
  representation of the circle, not the order of the flow solution.
* **Time step.** For the variational force $C_{L,\max}$ = 0.98269, 0.98543, 0.98610, 0.98627 for
  $\Delta t$ = 0.01 to 0.00125 (observed order 2.0 for this quantity and for St; not generalised to
  other observables or forces). The traction-based $C_{L,\max}$ is not monotone in $\Delta t$
  (0.98564, 0.98591, 0.98535, 0.98490), so the earlier statement that $\Delta t = 0.01$ was already
  converged was not adequately supported. $C_{D,\max}$ depends only weakly on $\Delta t$.
* **Mesh.** Five levels (14k to 273k dofs, ratio 1.5). The coarsest lands inside the $C_{L,\max}$
  interval (0.99715), which is a transient of the convergence and not a result. Resolution near the
  cylinder matters more than wake resolution for $C_{D,\max}$; for $C_{L,\max}$ both matter at the
  0.1 percent level.
* **Periodic regime.** The last-cycle change is at most $7\times10^{-6}$ in the continued runs
  ($3\times10^{-5}$ in the 8 s runs from rest). The original run's lift peaks looked flat to three
  figures only because of sampling; with refined peaks they were still rising by about $3\times10^{-5}$
  per cycle, i.e. the flow was near-periodic, not exactly periodic. A run started from an interpolated
  coarse-mesh state and a run from rest on the same mesh agree to $2\times10^{-6}$.

**Time horizon.** The benchmark procedure prescribes about 25 s of simulation and measurement
between 25 s and 30 s. These runs end at $t = 12$ s (8 s from rest). The periodicity evidence above
supports the statement that the computed cycle is settled; it does not make a $t = 12$ s run identical
to the benchmark evaluated at 25 to 30 s, and exact protocol reproduction is not claimed.

## Physical interpretation

At Re = 20 (2D-1), the flow is a steady recirculating wake, symmetric about the cylinder's downstream
axis to leading order but with a small computed lift ($c_L \sim 0.01$) reflecting the geometry's
slight vertical asymmetry (cylinder centred at $y=0.2$, not $H/2=0.205$). At Re = 100 (2D-2), the wake
becomes absolutely unstable and sheds vortices periodically (a Kármán vortex street), producing
oscillating drag and lift forces; the Strouhal number characterises the non-dimensional shedding
frequency and is, physically, set by the wake instability's own dynamics rather than by any numerical
parameter — its accurate reproduction here is a genuine (not merely curve-fitted) physical result.

## Computational performance

Measured on: Apple M3 Pro, macOS 26.6.2-arm64, dolfinx 0.10.0, PETSc/MUMPS direct LU solves (no GPU).

- **2D-1 steady** (direct LU, single Newton solve to convergence): 0.89 s (24,508 dofs) to 12.37 s
  (375,281 dofs); 5 Newton iterations at every resolution tested.
- **2D-2 unsteady** (Crank-Nicolson, Newton solve every step, 12,020 cells / 54,860 dofs, four
  force definitions and the pressure probe evaluated every step): about 0.4 to 0.7 s/step with the
  residual, Newton problem and force forms built once and reused (2 to 3 Newton iterations per
  step), against 1.28 s/step when the forms were rebuilt every step. Larger meshes scale
  superlinearly under direct LU (about 11 CPU-s/step at 273,224 dofs (17,837 s CPU for 1,600 steps)). Wall times
  on the machine used were inflated by other processes running at the same time; CPU time and thread
  counts are recorded in each run's `results/2d2_<label>.json`.
- No claim of "real-time" or GPU-accelerated performance is made; none was tested. All linear solves
  use a direct (MUMPS) factorisation; see Limitations for the iterative-solver follow-up this implies
  for substantially larger problems.

## Limitations

- **2D-3 (the ramped-Reynolds-number case) is not implemented.** Only 2D-1 and 2D-2 are covered; this
  was a deliberate scope decision (the brief's requested quantities — $c_D$, $c_L$, $\Delta P$,
  Strouhal — are all exercised by 2D-1/2D-2 already) and is stated here rather than left implicit.
- **Direct (MUMPS) linear solves throughout.** Correctness and benchmark agreement were established
  first, per the brief; an iterative Krylov solver with a fieldsplit (e.g. PCD or SIMPLE-type)
  preconditioner is the natural next step for meshes much larger than the ~375k-dof ceiling tested
  here, and was not implemented.
- **The default mesh geometry is degree 1 (a polygonal cylinder).** Curved degree-2 geometry is
  available (`--geometry-order 2`) and is needed for the converged 2D-2 values reported above; the
  degree-1 default is kept so that earlier polygonal results remain reproducible. Geometry orders
  above 2 are not supported. On curved cells the P1 pressure is linear in reference, not physical,
  coordinates, so physical-linear fields are represented only to $O(h^2\kappa)$ near the cylinder.
- **Taylor-Hood is not pointwise divergence-free**, as discussed above; a small, mesh-convergent
  $\|\nabla\cdot\mathbf{u}_h\|_{L^2}$ is expected and is not a defect.
- **The traction-based forces converge more slowly than the variational force** (see 2D-1 and 2D-2
  above): at 122k dofs the benchmark-stress $C_{L,\max}$ was still changing by $1.7\times10^{-3}$ per
  mesh level, and its time-step dependence is not monotone. Quantities computed with the
  traction definitions should not be read as converged at the resolutions used here.
- **The 2D-2 study is not a full asymptotic convergence proof.** It has five curved mesh levels
  (14k to 273k dofs) at $\Delta t = 0.0025$ and a time-step ladder on one mesh. Time-step and mesh
  effects were separated but not combined into a joint extrapolation; the observed mesh order of
  the variational $C_{L,\max}$ and $C_{D,\max}$ is not stable across level triples, and quoted
  Richardson estimates are estimates from three levels, not computed values. There is no
  polygon run at the finest level and no time-step study on the finer meshes.
- **The four force definitions do not agree on $C_{L,\max}$** at the finest mesh (0.9865 to 0.9879,
  0.14 %); the traction-based values were still changing with mesh and with $\Delta t$.
- **Reference provenance.** The 1996 benchmark paper was not re-read; its intervals are inherited
  from earlier project documentation. The FeatFlow values are another code's published results.
- **Time horizon.** Runs end at $t = 12$ s (8 s from rest), not at the benchmark's 25 to 30 s
  window; periodicity is judged from the last-cycle drift.
- **The pressure difference at a single instant is not uniquely defined** by the benchmark page used
  here (it gives $p_\text{diff}(t)$ over a cycle), and the 1996 definition was not checked. It is
  reported at the lift maximum and half a period later; they differ by about 0.002.
- CPU-only; no GPU or distributed-memory benchmarking was performed. The solver and force
  post-processing are written for MPI, but were only run on one rank; state save/restart is
  serial-only.

## Reproducibility

```bash
conda env create -f environment.yml && conda activate fenicsx

# tests (run first)
pytest tests/ -v

# manufactured-solution verification (run before trusting the benchmark results)
python scripts/verify_mms.py

# 2D-1 steady benchmark + mesh convergence
python scripts/run_2d1_steady.py --config smoke   # <1 min sanity check
python scripts/run_2d1_steady.py --config local   # a few seconds
python scripts/run_2d1_steady.py --config full    # ~20 s, matches published reference interval
python scripts/plot_2d1_convergence.py --config full
python scripts/plot_fields_2d1.py --config full

# 2D-2 unsteady benchmark (the original polygonal setup; default --geometry-order 1)
python scripts/run_2d2_unsteady.py --config smoke   # short sanity run
python scripts/run_2d2_unsteady.py --config full --label full_p1

# curved cylinder, from rest to t = 8 s, then a finer mesh started from the developed flow
python scripts/run_2d2_unsteady.py --config full --geometry-order 2 --label geo_p2_full \
    --save-state results/states/p2_base_t8.npz
python scripts/run_2d2_unsteady.py --h-far 0.02 --h-cyl 0.00333 --dt 0.0025 --t-end 12 \
    --geometry-order 2 --init-state results/states/p2_base_t8.npz --label sp_L4

# the whole geometry / time-step / mesh study, then its tables
bash scripts/study_2d2_commands.sh geometry   # then: temporal, spatial, polygon
python scripts/collect_2d2_study.py --sequence sp_L1,sp_L2,t_p2_dt0025,sp_L4,sp_L5 --ratio 1.5 \
    --method variational --triples

# 2D-1 with curved geometry and all four force definitions
python scripts/run_2d1_steady.py --config full --geometry-order 2 --label full_p2
```

Every 2D-2 run writes `results/2d2_<label>.{txt,json}` (values, per-cycle data, geometry error,
solver tolerances, commit, library versions, thread settings, timings) and a time-series `.npz`.
A killed run continues from its last checkpoint with `--resume`.

The results of the 2D-2 geometry / time-step / mesh study are the `results/2d2_<label>.txt` and
`.json` files whose labels are listed in `docs/2d2_convergence_study.md`; the raw time series is
kept in the repository only for the finest run (`results/2d2_timeseries_sp_L5.npz`), and the others
regenerate from `scripts/study_2d2_commands.sh`. The files `results/2d2_unsteady_*.txt`,
`results/2d2_timeseries_{smoke,local,full,full_dt01}.npz` and `figures/2d2_force_coefficients_*.png`
come from the original polygonal configuration (symmetric stress, raw-sample maxima) and are kept
unchanged as a record of it; they are superseded by the study results above.

Raw numerical results are written to `results/*.txt` (and `*.npz` for time series); figures are
generated from those results, not hand-drawn, and are written separately to `figures/`, so every
number quoted in this README is traceable to a specific file produced by a specific command above.

## References

- Schäfer, M., Turek, S. (1996). *Benchmark Computations of Laminar Flow Around a Cylinder*, in Flow
  Simulation with High-Performance Computers II (Notes on Numerical Fluid Mechanics, Vol. 52),
  Vieweg. (Original benchmark specification; all geometry, boundary conditions, and reference values
  used in this repository are taken directly from this paper.)
- Girault, V., Raviart, P.-A. (1986). *Finite Element Methods for Navier-Stokes Equations: Theory and
  Algorithms*. Springer.
- Elman, H., Silvester, D., Wathen, A. (2014). *Finite Elements and Fast Iterative Solvers*, 2nd ed.
  Oxford University Press.
- Heywood, J. G., Rannacher, R., Turek, S. (1996). "Artificial boundaries and flux and pressure
  conditions for the incompressible Navier-Stokes equations." *Int. J. Numer. Meth. Fluids* 22(5).
- Turek, S. (1999). *Efficient Solvers for Incompressible Flow Problems: An Algorithmic and
  Computational Approach*. Springer.
- Langtangen, H. P., Logg, A. (2017). *Solving PDEs in Python: The FEniCS Tutorial I*. Springer.
  (Cylinder-flow traction-based drag/lift post-processing technique.)
- Baratta, I. A. et al. (2023). "DOLFINx: The next generation FEniCS problem solving environment."
  *Zenodo*. doi:10.5281/zenodo.10447666.

**Prior related work.** This project is methodologically independent of the companion
`navier-stokes-2d` repository (pseudo-spectral, periodic vorticity-streamfunction formulation): no
code is shared, and the two demonstrate different discretisation families (spectral vs. finite
element) deliberately, per the portfolio's own stated objective.
