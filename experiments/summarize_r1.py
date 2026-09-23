"""Summary of a phase-16 R1 log: per variant the seeds, mean, sd, and Welch's t against a reference.

    python experiments/summarize_r1.py results/phase16_r1/r1.log [--reference r1_baseline_l2]

Reads the `[diag]` lines `experiments/diag_short.py` prints, so it works on a finished or a running
log. It prints the decision rules of `experiments/run_r1.sh` with the measured differences filled in;
it never decides anything by itself.
"""

import argparse
import math
import re
import sys
from collections import defaultdict

FIELD = re.compile(r"(\w+)=([^\s|]+)")
RULES = [
    ("R1.1 pooled - eppm    >= +3, t > 3", "r1_pooled", "r1_eppm", "the class-slot correlation of D-01 is a main cause"),
    ("R1.2 |eppms - vippem| <= 2", "r1_eppms", "r1_vippem", "EPPM-S reaches the reference head (route A)"),
    ("R1.3 eppms - eppm     >= +3", "r1_eppms", "r1_eppm", "the cause is in the additions of Eq.19-21"),
    ("R1.4 vippem - eppm    >= +3", "r1_vippem", "r1_eppm", "VIP-Seg's stage is ahead (route B if EPPM-S is not)"),
]


def parse(path):
    """variant -> list of (seed, valid mIoU), in log order."""
    runs = defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("[diag]"):
                continue
            fields = dict(FIELD.findall(line))
            if "variant" in fields and "valid_miou" in fields:
                runs[fields["variant"]].append((int(fields.get("seed", -1)), float(fields["valid_miou"])))
    return runs


def stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var), n


def welch(a, b):
    """(difference, standard error, t) of two samples; nan when either has fewer than two runs."""
    (ma, sa, na), (mb, sb, nb) = stats(a), stats(b)
    if na < 2 or nb < 2:
        return ma - mb, float("nan"), float("nan")
    se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
    return ma - mb, se, (ma - mb) / se if se else float("inf")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--reference", default="r1_baseline_l2", help="variant every row is also compared with")
    args = p.parse_args(argv)
    runs = parse(args.log)
    if not runs:
        print(f"no [diag] lines in {args.log}")
        return 1
    ref = [v for _, v in runs.get(args.reference, [])]
    print(f"{'variant':20s} {'seeds':>28s} {'mean':>8s} {'sd':>7s} {'vs ' + args.reference:>22s}")
    for variant, values in runs.items():
        vals = [v for _, v in values]
        mean, sd, n = stats(vals)
        seeds = " ".join(f"{v:.4f}" for v in vals)
        against = ""
        if ref and variant != args.reference:
            d, se, t = welch(vals, ref)
            against = f"{d:+.4f} (t={t:+.2f})" if not math.isnan(t) else f"{d:+.4f}"
        print(f"{variant:20s} {seeds:>28s} {mean:8.4f} {sd:7.4f} {against:>22s}   n={n}")
    print("\ndecision rules of experiments/run_r1.sh:")
    for label, a, b, meaning in RULES:
        va, vb = [v for _, v in runs.get(a, [])], [v for _, v in runs.get(b, [])]
        if not va or not vb:
            print(f"  {label:34s} -> not run yet ({a} or {b} missing)")
            continue
        d, se, t = welch(va, vb)
        print(f"  {label:34s} -> {d * 100:+.2f} points, t={t:+.2f}   [{meaning}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
