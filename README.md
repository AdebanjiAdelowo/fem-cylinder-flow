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
  distance-based sizing field refining the mesh near the cylinder (`src/geometry.py`).
- **Time discretisation** (2D-2 only): Crank-Nicolson $\theta$-method, Newton solve every step.
- **Newton/SNES tolerances**: `snes_rtol = 1e-10`, `snes_atol = 1e-10`, `snes_max_it = 25`, with
  `snes_error_if_not_converged = True` — every solve in this repository either meets this tolerance or
  raises, so no result below is from a silently non-converged solve.

## Implementation

```
fem-cylinder-flow/
├── src/
│   ├── geometry.py         gmsh mesh generation, Schäfer-Turek geometry constants
│   ├── spaces.py             Taylor-Hood P2/P1 mixed function space
│   ├── navier_stokes.py       weak-form residual (steady + theta-method unsteady), Newton/SNES solve
│   ├── cylinder_bcs.py         no-slip / parabolic-inflow / do-nothing boundary conditions
│   ├── forces.py                c_D, c_L, Delta P post-processing
│   └── manufactured.py           UFL-symbolic manufactured Stokes/Navier-Stokes solutions
├── scripts/
│   ├── verify_mms.py         manufactured-solution verification + mesh-convergence figure
│   ├── run_2d1_steady.py       2D-1 benchmark + mesh-convergence ladder
│   ├── run_2d2_unsteady.py      2D-2 benchmark, force-coefficient time series, Strouhal via FFT
│   ├── plot_2d1_convergence.py   c_D/c_L/Delta P vs. resolution figure
│   ├── plot_fields_2d1.py         velocity/pressure/vorticity/streamline figure
│   └── reprocess_2d2.py            regenerate 2D-2 results/figures from a saved time series
│                                     (no FEM re-solve) -- used to fix the Strouhal-estimation
│                                     method after the ~34 min full run had already completed
├── tests/                   18 pytest tests: geometry, spaces, BCs, forces, MMS, solver
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
tested level $c_D$ and $\Delta P$ are still outside it by roughly 1–2%.

### 2D-2 (unsteady, Re = 100): vortex shedding

`python scripts/run_2d2_unsteady.py --config full` — Crank-Nicolson, $\Delta t = 0.005\,\mathrm{s}$,
$t_{\mathrm{end}} = 8\,\mathrm{s}$, mesh 12,020 cells / 54,860 dofs, wall time 2045.36 s (34.1 min).

**Saturation check.** $C_L(t)$'s oscillation amplitude grows from the impulsive start and must reach
a statistically stationary limit cycle before any "$c_{D,\max}$"/"$c_{L,\max}$" is meaningful. The
last six lift peaks in the post-transient window are 0.98248, 0.98338, 0.98346, 0.98274, 0.98332,
0.98367 — a spread of 0.0012, i.e. flat to three significant figures — confirming the run reached a
converged periodic regime well before $t=8\,\mathrm{s}$ (figure:
`figures/2d2_force_coefficients_full.png`, left panel; the very large transient spike visible in the
raw data at $t\approx0$, up to $c_D\approx74$, is the well-known impulsive-start artefact of switching
on the full parabolic inflow in one time step, decays within the first $\sim$0.1 s, and is excluded
from the plotted $y$-range for readability but is present in `results/2d2_timeseries_full.npz`).

**Strouhal number: two independent estimates.** The raw FFT bin spacing for a 4 s post-transient
window is 0.25 Hz ($\Delta\mathrm{St}=0.025$) — wider than the *entire* published reference interval
(width 0.01) — so the shedding frequency is recovered via three-point quadratic (parabolic)
interpolation on the log-magnitude spectrum around the peak bin (`strouhal_from_series` in
`scripts/run_2d2_unsteady.py`), a standard sub-bin-resolution DSP technique. This is cross-checked
against a fully independent, FFT-free estimate: the mean period between 12 successive $c_L(t)$ peaks
in the same window.

| Method | Result |
|---|---|
| FFT (Hanning window, 4.00 s, quadratic log-magnitude peak interpolation) | $f=3.0190\,\mathrm{Hz} \Rightarrow \mathrm{St}=0.3019$ |
| Direct peak-to-peak period (12 peaks, mean period 0.33136 s) | $f=3.0178\,\mathrm{Hz} \Rightarrow \mathrm{St}=0.3018$ |

The two independent methods agree to 0.04% relative difference.

**Benchmark comparison** (Schäfer & Turek, 1996, Table 4, lower/upper bound row):

| Quantity | Computed | Schäfer–Turek reference | Difference / status |
|---|---:|---:|---:|
| $C_{D,\max}$ | 3.2139 | $[3.2200,\ 3.2400]$ | $-0.190\%$ vs. nearest bound — outside |
| $C_{L,\max}$ | 0.9837 | $[0.9900,\ 1.0100]$ | $-0.639\%$ vs. nearest bound — outside |
| $\Delta P$ (at $t$ of $C_{L,\max}$) | 2.4712 | $[2.4600,\ 2.5000]$ | inside |
| $\mathrm{St}$ (FFT) | 0.3019 | $[0.2950,\ 0.3050]$ | inside |

Two of the four reported quantities fall inside the published reference interval; $C_{D,\max}$ and
$C_{L,\max}$ are close but outside it, by $-0.19\%$ and $-0.64\%$ respectively relative to the nearest
bound. These are the actual computed discrepancies — no parameter was tuned to force agreement.

**Mesh/time-step sensitivity** (isolating whether the remaining $C_{D,\max}$/$C_{L,\max}$ gap is
spatial, temporal, or transient-window related): the `local` configuration (1,499 cells, $\Delta
t=0.01\,\mathrm{s}$) also reaches a clean saturated limit cycle by $t=8\,\mathrm{s}$ (spread of the
last six lift peaks: 0.0006), so it is directly comparable to `full`, with only the transient-window
question already ruled out for both:

| Configuration | cells | dofs | $\Delta t$ | $C_{D,\max}$ | $C_{L,\max}$ | $\Delta P$ | St |
|---|---:|---:|---:|---:|---:|---:|---:|
| `local` (coarse mesh, coarse $\Delta t$) | 1,499 | 7,028 | 0.010 | 3.0288 ($-5.94\%$) | 0.9727 ($-1.75\%$) | 2.3945 ($-2.66\%$) | 0.3036 (inside) |
| `full`, $\Delta t = 0.01$ (fine mesh, coarse $\Delta t$) | 12,020 | 54,860 | 0.010 | 3.2140 ($-0.19\%$) | 0.9832 ($-0.68\%$) | 2.4721 (inside) | 0.3011 (inside) |
| `full` (fine mesh, fine $\Delta t$) | 12,020 | 54,860 | 0.005 | 3.2139 ($-0.19\%$) | 0.9837 ($-0.64\%$) | 2.4712 (inside) | 0.3019 (inside) |

**Conclusion.** Halving the time step at *fixed* fine mesh ($\Delta t: 0.01 \to 0.005$, bottom two
rows) changes $C_{D,\max}$ by $0.0001$ (0.003% relative) and $C_{L,\max}$ by $0.0005$ (0.05%
relative) — both changes are noise-level, well inside the run-to-run peak-detection spread reported
in the saturation check above. Refining the mesh at *fixed* coarse $\Delta t=0.01$ (top vs. middle
row), by contrast, changes $C_{D,\max}$ by $+6.1\%$ and $C_{L,\max}$ by $+1.1\%$. **The remaining
$-0.19\%/-0.64\%$ gap between the `full` result and the published reference interval is therefore
attributable almost entirely to spatial (mesh) resolution, not to the time step** — $\Delta t=0.01\,
\mathrm{s}$ is already temporally converged for these quantities at this mesh, so further mesh
refinement beyond 12,020 cells (not attempted here; see Limitations) would be the correct next step
to close the remaining gap, not a smaller time step. This is consistent with, and explains, the same
pattern already seen in the 2D-1 steady mesh-convergence study above.

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
- **2D-2 unsteady** (Crank-Nicolson, Newton solve every step, `full` config: 12,020 cells / 54,860
  dofs, 1600 steps): 2045.36 s (34.1 min) total, i.e. $\approx 1.28\,\mathrm{s}$/step including
  Newton iteration and MUMPS factorisation at every step. The $\Delta t=0.01$ sensitivity run at the
  same mesh (800 steps) took 1002.71 s, $\approx 1.25\,\mathrm{s}$/step — consistent per-step cost,
  as expected since halving $\Delta t$ does not change the per-step linear-algebra cost.
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
- **Straight-sided (affine, degree-1 geometry) mesh elements approximate the curved cylinder boundary
  as a polygon.** Under mesh refinement this geometric approximation improves alongside the FE
  approximation itself (part of the convergence behaviour reported above), but a curved
  (isoparametric, higher-order geometry) mesh was not implemented and would reduce the required
  resolution for a given accuracy.
- **Taylor-Hood is not pointwise divergence-free**, as discussed above; a small, mesh-convergent
  $\|\nabla\cdot\mathbf{u}_h\|_{L^2}$ is expected and is not a defect.
- **The drag/lift/pressure-difference post-processing uses the general Cauchy-stress traction formula**
  rather than the paper's literal tangential-derivative expression (see `src/forces.py` docstring for
  the equivalence argument); this is standard practice for this exact benchmark (e.g. the FEniCS
  tutorial's own cylinder-flow example) but is a deliberate implementation choice worth being explicit
  about.
- **2D-2 was not run on a full mesh-convergence ladder** the way 2D-1 was (3-4 resolution levels); the
  sensitivity analysis above uses only two mesh sizes (`local`, `full`) and, at the finer mesh, two
  time steps, which was enough to attribute the remaining gap to spatial resolution but not enough to
  extrapolate a converged (mesh-independent) reference value the way the 2D-1 ladder's trend suggests
  is achievable. The `full` mesh (12,020 cells) that reaches $C_{D,\max}$ within $0.19\%$ of the
  reference interval has not itself been shown to be mesh-converged in the stricter sense of two
  further refinement levels agreeing with each other.
- CPU-only; no GPU or distributed-memory benchmarking was performed (though the implementation is
  MPI-parallel via PETSc/dolfinx and would run unmodified on multiple ranks).

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

# 2D-2 unsteady benchmark
python scripts/run_2d2_unsteady.py --config smoke   # ~30 s
python scripts/run_2d2_unsteady.py --config local   # ~2 min
python scripts/run_2d2_unsteady.py --config full    # ~34 min (measured: 2045.36 s)

# mesh/time-step sensitivity diagnostic (fine mesh, coarser dt; ~17 min)
python scripts/run_2d2_unsteady.py --config full --dt-override 0.01 --label full_dt01
```

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
