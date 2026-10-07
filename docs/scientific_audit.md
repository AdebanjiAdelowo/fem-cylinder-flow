# Scientific Audit — 2D Incompressible Flow Past a Cylinder

## Verdict summary

| Component | Verdict |
|---|---|
| Governing problem | **Supported.** Standard 2D incompressible Navier–Stokes cylinder benchmark. |
| Taylor–Hood P2/P1 FEM | **Appropriate.** Inf-sup stable mixed pair. |
| Newton/UFL Jacobian | **Appropriate and well verified.** |
| Manufactured-solution verification | **Strong.** Exactly representable polynomial case reaches roundoff; smooth trigonometric case shows expected velocity convergence. |
| Cylinder geometry treatment | **Strongly qualified/controlled.** Linear and quadratic geometry are explicitly separated and measured. |
| 2D-1 benchmark | **Supported.** Finest curved-mesh results approach the high-order reference; variational reaction is especially close. |
| 2D-2 benchmark | **Supported with qualifications.** Period/frequency is well converged; force extrema remain slightly below inherited interval bounds on the audited mesh. |
| Exact benchmark-instant pressure comparison | **Not established.** Repository correctly reports the pressure quantity at identified lift extrema/phase rather than pretending the original benchmark instant was independently reconstructed. |
| 2D-3 | **Not implemented.** |
| Wrong problem/method? | **No.** No fundamental problem-formulation or numerical-method defect identified in this pass. |

## 1. Mathematical problem

The repository solves
[
\partial_t u-\nu\Delta u+(u\cdot\nabla)u+\nabla p=f,qquad \nabla\cdot u=0
]
with the DFG channel/cylinder geometry, prescribed no-slip/inflow conditions and a natural outlet condition. The steady Re=20 and unsteady Re=100 cases are standard benchmark problems.

The outlet condition is a modelling/numerical boundary choice, not an exact representation of an infinitely long downstream domain. At the benchmark Reynolds numbers and geometry it is a conventional choice, but it should not be generalised to arbitrary severe backflow problems.

## 2. Discretisation

Continuous P2 velocity / P1 pressure Taylor–Hood elements are an appropriate stable mixed finite-element pair. The nonlinear residual uses the convective form directly, with exact symbolic Jacobian differentiation through UFL and a direct linear solve inside Newton.

For the unsteady case the momentum spatial operator is theta-blended while pressure/continuity remain at the new time level. The reported primary 2D-2 studies use Crank–Nicolson (`theta=0.5`).

**Classification:** right problem, right primary method.

## 3. Manufactured-solution verification

Two complementary checks are particularly useful:

- an exactly representable polynomial velocity/pressure solution is recovered to floating-point roundoff;
- a trigonometric solution produces approximately third-order velocity L2 and second-order velocity H1 convergence, consistent with P2 approximation.

This tests the PDE assembly independently of the cylinder benchmark values and is stronger evidence than benchmark agreement alone.

## 4. Geometry and forces

A major strength of the repository is that it does not silently treat a polygonal cylinder as an exact circle. It records represented perimeter/area/radius errors and supports quadratic geometry.

Four force evaluations are retained:

- benchmark/Laplacian traction;
- symmetric Cauchy stress;
- tangential-derivative form;
- discrete variational reaction.

These need not agree exactly on a finite Taylor–Hood mesh because discrete velocity is not pointwise divergence free and boundary geometry/quadrature differ. Their convergence toward one another is therefore informative.

## 5. 2D-1

On the finest reported quadratic-geometry mesh, the primary Laplacian values are approximately

- (c_D=5.57829);
- (c_L=0.010489);
- (Delta p=0.11751).

They are close to the high-order reference recorded by the repository. The variational reaction is even closer for drag/lift. The project correctly presents the spread among force definitions rather than hiding it.

**Verdict:** benchmark reproduction is well supported at the reported resolution.

## 6. 2D-2

The quadratic-geometry full run reports a stable shedding period around 0.33157 and (St\approx0.30160), with cycle/FFT/harmonic-fit estimates agreeing closely. This is strong evidence that the periodic frequency has settled.

The primary force extrema are (c_{D,max}\approx3.21758) and (c_{L,max}\approx0.98587). They sit slightly below the repository's inherited comparison-interval lower bounds, while remaining close to a finer FeatFlow reference quoted for orientation.

This must be described as **near-benchmark with residual spatial/time/post-processing sensitivity**, not as exact convergence to every interval.

## 7. Pressure difference and benchmark timing

The original benchmark's exact reporting instant is not reconstructed independently here. The repository evaluates pressure difference at a clearly identified lift maximum and half-period phase. That is acceptable provided the quantity is not mislabeled as the exact original benchmark-instant value.

## 8. Important limitation in the test philosophy

One force regression test asserts that, on a solved coarse flow, the variational force is the closest of the four definitions to a hard-coded spectral benchmark value. This is useful as a regression guard but is **not independent verification** of the force implementation: it contains external benchmark knowledge in the assertion. The analytical pressure-gradient/divergence-theorem force tests are the genuinely independent implementation checks.

No code change is required for this distinction, but it should be remembered when weighing evidence.

## 9. Missing scope

DFG 2D-3 is not implemented. The project therefore supports claims about 2D-1 and 2D-2 only.

## Final classification

**Supported project with qualifications.** The PDE, finite-element spaces, nonlinear solve and verification strategy are appropriate. No wrong-problem or wrong-method correction is warranted. The main scientific caution is to distinguish “close to/reference-consistent” from “fully converged to every published interval,” particularly for 2D-2 force extrema and benchmark timing.
