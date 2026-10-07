# Handoff: cylinder-flow benchmark verification

## Current scientific status
The standard 2D DFG/Schäfer-Türek cylinder benchmark is correctly posed and P2/P1 Taylor-Hood with Newton solution is appropriate. MMS verification is strong. The 2D-1 result is well supported. The 2D-2 Strouhal result is close to reference behavior, while force extrema retain small sensitivity/discrepancy across geometry, force extraction and temporal/spatial resolution. 2D-3 is not implemented.

## Local work
Run the full tests, MMS, 2D-1 smoke/local/full as feasible, then 2D-2 temporal/spatial/geometry sensitivity studies. Preserve all four force formulations and explain finite-mesh differences rather than selecting whichever matches a reference best. Confirm phase selection for pressure difference and lift extrema. Do not use the hard-coded reference-comparison regression as independent validation; analytical force tests and convergence studies are the independent evidence. Record mesh/geometry order, dt, solver tolerances and force definition with every headline value.

## Claude prompt
Continue the scientific audit of fem-cylinder-flow. Read this handoff, docs/scientific_audit.md, configs, MMS tests, geometry/force tests and committed 2D-1/2D-2 result files. Run the complete tests and verification scripts in the existing FEniCSx environment. Reproduce MMS first, then 2D-1, then 2D-2 resolution studies. Quantify changes in cD, cL, pressure difference, period and Strouhal with mesh, dt, geometry order and force definition. Treat small discrepancies as numerical/post-processing sensitivity unless evidence shows a bug. Do not claim 2D-3. Do not tune a force formula to a benchmark. Show test results and old/new numerical comparisons before commits; do not merge or force-push.