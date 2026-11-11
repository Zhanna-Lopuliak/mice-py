"""Print the layer-2 numbers for every method and input (for the baseline).

Run from the repository root:

    uv run python -m tests.test_parity._l2_report

For each method and input: smallest T1 p-value over cells, largest |z| of T2
(cell mean and log-variance) and T3 (coefficient mean and log-variance), and
for the tree methods the weaker measures used by test_l2_trees. T1 compares
donor counts, for the norm family (continuous kind) the counts in R's decile
bins, for logreg, logreg.boot and polyreg (category kind) the level counts;
their T3 is on the coefficients of the level indicators and they have no T2. A crash
is reported with its exception. Pass rules: p >= 1e-7, |z| <= 5.5.
"""

import warnings

from . import _l2_harness as H

ROWS = [
    ("pmm", "F1", {}),
    ("pmm", "F2", {}),
    ("pmm", "F3", {}),
    ("pmm", "F4", {}),
    ("midastouch", "F1", {}),
    ("midastouch", "F2", {}),
    ("midastouch", "F4", {}),
    ("cart", "F1", {}),
    ("cart", "F2", {}),
    ("cart", "F3", {}),
    ("cart", "F4", {}),
    ("cart", "F1", {"bootstrap": True}),
    ("rf", "F1", {}),
    ("rf", "F2", {}),
    ("rf", "F3", {}),
    ("rf", "F4", {}),
    ("sample", "F1", {}),
    ("sample", "F3", {}),
    ("norm", "F1", {}),
    ("norm", "F4", {}),
    ("norm.nob", "F1", {}),
    ("norm.boot", "F1", {}),
    ("logreg", "F5", {}),
    ("logreg", "F6", {}),
    ("logreg.boot", "F5", {}),
    ("polyreg", "F3", {}),
]


def fmt(v, spec):
    return "-" if v != v else format(v, spec)  # NaN -> "-"


def main():
    warnings.simplefilter("ignore")
    print(f"{'method':<12}{'input':<6}{'T1 min p':>10}{'T2 |z|':>9}{'T3 |z|':>9}  other [arguments]")
    for method, name, kwargs in ROWS:
        args = ",".join(f"{k}={v}" for k, v in kwargs.items()) or "defaults"
        try:
            inp, r_sum, py_sum = H.run_python(method, name, **kwargs)
        except Exception as exc:  # report crashes, do not stop
            print(f"{method:<12}{name:<6}  {type(exc).__name__}: {exc} [{args}]")
            continue
        comp = H.compare(r_sum, py_sum, inp)
        kind = H.KINDS[py_sum["kind"]]
        other = []
        if py_sum["kind"] != "donor":
            other.append(f"T1 on {kind.t1_label}")
        if py_sum.get("n_not_donor"):
            other.append(f"{py_sum['n_not_donor']} imputations not {kind.not_recorded}")
        if method in ("cart", "rf"):
            if inp.is_factor:
                other.append(f"level-prop diff {H.weak_factor(r_sum, py_sum):.3f}")
            elif "cell" in py_sum:
                dev, ratio = H.weak_numeric(r_sum, py_sum)
                other.append(f"mean dev {dev:.3f} R-sd, var ratio {ratio:.3f}")
        if method == "cart" and not inp.is_factor:
            other.append(f"donor-set mismatches {len(H.donor_set_mismatches(r_sum, py_sum))}")
        print(
            f"{method:<12}{name:<6}{comp.t1_min_p:>10.3g}"
            f"{fmt(comp.t2_max_z, '.2f'):>9}{fmt(comp.t3_max_z, '.2f'):>9}  " + "; ".join(other + [f"[{args}]"])
        )


if __name__ == "__main__":
    main()
