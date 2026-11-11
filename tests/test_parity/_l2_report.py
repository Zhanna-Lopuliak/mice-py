"""Print the layer-2 numbers for every method and input (for the baseline).

Run from the repository root:

    uv run python -m tests.test_parity._l2_report

For each method and input: smallest T1 p-value over cells, largest |z| of T2
(cell mean and log-variance) and T3 (coefficient mean and log-variance), and
for the tree methods the weaker measures used by test_l2_trees. A crash is
reported with its exception. Pass rules: p >= 1e-7, |z| <= 5.5.
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
    ("cart", "F1", {"bootstrap": False, "cp": 1e-4, "min_samples_leaf": 5}),
    ("rf", "F1", {}),
    ("rf", "F2", {}),
    ("rf", "F3", {}),
    ("rf", "F4", {}),
    ("sample", "F1", {}),
    ("sample", "F3", {}),
]


def fmt(v, spec):
    return "-" if v != v else format(v, spec)  # NaN -> "-"


def main():
    warnings.simplefilter("ignore")
    print(f"{'method':<11}{'input':<6}{'T1 min p':>10}{'T2 |z|':>9}{'T3 |z|':>9}  other [arguments]")
    for method, name, kwargs in ROWS:
        args = ",".join(f"{k}={v}" for k, v in kwargs.items()) or "defaults"
        try:
            inp, r_sum, py_sum = H.run_python(method, name, **kwargs)
        except Exception as exc:  # report crashes, do not stop
            print(f"{method:<11}{name:<6}  {type(exc).__name__}: {exc} [{args}]")
            continue
        comp = H.compare(r_sum, py_sum, inp)
        other = []
        if py_sum["n_not_donor"]:
            other.append(f"{py_sum['n_not_donor']} imputations not observed values")
        if method in ("cart", "rf"):
            if inp.is_factor:
                other.append(f"level-prop diff {H.weak_factor(r_sum, py_sum):.3f}")
            elif "cell" in py_sum:
                dev, ratio = H.weak_numeric(r_sum, py_sum)
                other.append(f"mean dev {dev:.3f} R-sd, var ratio {ratio:.3f}")
        if method == "cart" and not inp.is_factor:
            other.append(f"donor-set mismatches {len(H.donor_set_mismatches(r_sum, py_sum))}")
        print(
            f"{method:<11}{name:<6}{comp.t1_min_p:>10.3g}"
            f"{fmt(comp.t2_max_z, '.2f'):>9}{fmt(comp.t3_max_z, '.2f'):>9}  "
            + "; ".join(other + [f"[{args}]"])
        )


if __name__ == "__main__":
    main()
