"""Tabulate 2D-2 runs (results/2d2_<label>.json) and estimate observed convergence orders.

    python scripts/collect_2d2_study.py --labels a,b,c                      # one row per run
    python scripts/collect_2d2_study.py --labels a,b,c --methods            # all force definitions
    python scripts/collect_2d2_study.py --sequence l1,l2,l3 --ratio 1.5     # observed order, coarse->fine

Observed order: for three consecutive levels with constant refinement ratio r and monotone
values Q1, Q2, Q3,  p = ln((Q2 - Q1) / (Q3 - Q2)) / ln(r). A Richardson estimate of the limit,
Q* ~ Q3 + (Q3 - Q2) / (r^p - 1), is printed only when the sequence is monotone and 0.5 <= p <= 4;
it is an ESTIMATE from the last three levels, not a computed result, and is labelled as such.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUANTITIES = ("cd_max", "cl_max", "cl_min", "period", "dp_at_cl_max")


def load(label):
    return json.loads((ROOT / "results" / f"2d2_{label}.json").read_text())


def st_of(rec, method):
    return rec["methods"][method]["strouhal"]["cycle_period"]


def table(labels, methods_mode):
    for lab in labels:
        r = load(lab)
        m, prim = r["meta"], r["primary_force"]
        names = list(r["methods"]) if methods_mode else [prim]
        for name in names:
            v = r["methods"][name]["values"]
            d = r["methods"][name]["drift"]
            sp = r["methods"][name]["spread"]
            g = m["geometry"]
            print(f"{lab:<22}{name:<12}{m['n_cells']:>7}{m['n_dofs']:>8} o{m['geometry_order']} dt={m['dt']:<8g} "
                  f"area_err={g['area_rel_err']:+.1e} | cD,max={v['cd_max']:.5f} cL,max={v['cl_max']:.5f} "
                  f"cL,min={v['cl_min']:.5f} St={st_of(r, name):.5f} dP={v['dp_at_cl_max']:.4f} | "
                  f"drift(cL,max)={d['cl_max']:.1e} spread4={sp['cl_max']:.1e} "
                  f"periodic={r['methods'][name]['periodic']} cycles={r['methods'][name]['n_cycles']}")


def observed_order(values, ratio):
    q1, q2, q3 = values[-3:]
    d12, d23 = q2 - q1, q3 - q2
    if d12 == 0 or d23 == 0 or d12 * d23 < 0:
        return None, None, "non-monotone"
    p = math.log(abs(d12 / d23)) / math.log(ratio)
    if not 0.5 <= p <= 4.0:
        return p, None, "order outside [0.5, 4]: no extrapolation"
    return p, q3 + d23 / (ratio ** p - 1.0), "monotone"


def all_triples(labels, ratio, method):
    """Observed order and (where justified) Richardson estimate for EVERY consecutive triple of
    levels, so that no subset of levels is chosen to obtain a preferred order."""
    recs = [load(l) for l in labels]
    print(f"every consecutive triple, primary force={method}, ratio {ratio}")
    for q in QUANTITIES + ("st",):
        vals = [st_of(r, method) if q == "st" else r["methods"][method]["values"][q] for r in recs]
        cells = []
        for i in range(len(vals) - 2):
            p, est, note = observed_order(vals[i:i + 3], ratio)
            cells.append(f"L{i+1}-{i+3}: " + (f"p={p:.2f}" if p is not None else "p=-") +
                         (f" est={est:.5f}" if est is not None else f" ({note})"))
        print(f"{q:<14}" + "   ".join(cells))


def sequence(labels, ratio, method):
    recs = [load(l) for l in labels]
    print(f"observed convergence, primary force={method}, refinement ratio {ratio}, levels: {', '.join(labels)}")
    print(f"{'quantity':<14}" + "".join(f"{l:>14}" for l in labels) + f"{'diff(last)':>13}{'order p':>9}{'Richardson est.':>17}  note")
    for q in QUANTITIES + ("st",):
        vals = [st_of(r, method) if q == "st" else r["methods"][method]["values"][q] for r in recs]
        diff = vals[-1] - vals[-2]
        if len(vals) >= 3:
            p, est, note = observed_order(vals, ratio)
        else:
            p, est, note = None, None, "need 3 levels"
        print(f"{q:<14}" + "".join(f"{v:>14.5f}" for v in vals) + f"{diff:>+13.2e}"
              f"{('%.2f' % p) if p is not None else '-':>9}{('%.5f' % est) if est is not None else '-':>17}  {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--methods", action="store_true", help="one row per force definition")
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--method", default="laplacian")
    ap.add_argument("--triples", action="store_true", help="with --sequence: every consecutive triple")
    a = ap.parse_args()
    if a.labels:
        table(a.labels.split(","), a.methods)
    if a.sequence:
        sequence(a.sequence.split(","), a.ratio, a.method)
        if a.triples:
            all_triples(a.sequence.split(","), a.ratio, a.method)


if __name__ == "__main__":
    main()
