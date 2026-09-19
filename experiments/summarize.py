"""Phase-14 results next to the paper's Tables 2, 4 and 5 (S3DIS, text modality), as markdown.

    python experiments/summarize.py [--save_dir log_phase14] [--protocol fixed100]

Reads the `eval_<best|last>_<protocol>.json` files written by experiments/phase14.py. mIoU in %.
"""

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import phase14  # noqa: E402

# (S0, S1, Avg) as printed [PAPER Tab.4] [PAPER Tab.5] [PAPER Tab.2, CascadeProto (Text)]
PAPER: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "T4": {"baseline": (82.72, 79.83, 81.28), "lma": (83.98, 80.99, 82.49), "gate": (85.34, 82.48, 83.91),
           "cascade": (87.89, 84.05, 85.97), "full": (88.53, 84.53, 86.53)},
    "T5": {"T1": (85.21, 81.34, 83.28), "T2": (86.43, 82.67, 84.55), "T3": (87.78, 83.91, 85.85),
           "T4": (88.53, 84.53, 86.53), "T5": (88.51, 84.50, 86.51), "T6": (88.37, 84.31, 86.34)},
    "T2": {"N2K1": (88.53, 84.53, 86.53), "N2K5": (88.57, 84.78, 86.68), "N3K1": (83.07, 79.04, 81.06),
           "N3K5": (79.95, 77.94, 78.95)},
}
TITLES = {"T4": "Table 4 - components (2-way 1-shot)", "T5": "Table 5 - cascade depth T (2-way 1-shot)",
          "T2": "Table 2 - CascadeProto (Text), all settings"}


def read_result(save_dir: str, run, which: str, protocol: str) -> Optional[float]:
    """mIoU in % from a finished evaluation, or None (missing, or a dry run)."""
    path = phase14.result_path(run, save_dir, which, protocol)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        result = json.load(f)
    return None if result.get("dry_run") else 100.0 * result["miou"]


def cell_values(save_dir: str, protocol: str, table: str, row: str, which: str) -> Dict[int, Optional[float]]:
    """{fold: mIoU} of the seed-0 run that fills (table, row)."""
    out = {0: None, 1: None}
    for run in phase14.all_runs():
        if (table, row) in run.tables and run.seed == 0:
            out[run.cvfold] = read_result(save_dir, run, which, protocol)
    return out


def fmt(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.2f}"


def fmt_delta(ours: Optional[float], paper: float) -> str:
    return "-" if ours is None else f"{ours - paper:+.2f}"


def table_markdown(save_dir: str, protocol: str, table: str) -> str:
    lines = [f"### {TITLES[table]}", "",
             "| Row | S0 best | S0 last | S1 best | S1 last | Avg best | Paper S0 / S1 / Avg | Avg diff (best) |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: |"]
    for row, (p0, p1, pavg) in PAPER[table].items():
        best, last = (cell_values(save_dir, protocol, table, row, w) for w in ("best", "last"))
        avg = None if None in (best[0], best[1]) else (best[0] + best[1]) / 2
        lines.append(f"| {row} | {fmt(best[0])} | {fmt(last[0])} | {fmt(best[1])} | {fmt(last[1])} | {fmt(avg)} | "
                     f"{p0:.2f} / {p1:.2f} / {pavg:.2f} | {fmt_delta(avg, pavg)} |")
    return "\n".join(lines)


def seed_markdown(save_dir: str, protocol: str) -> str:
    """Spread of the full model on S0 2-way 1-shot over seeds 0, 1, 2 (best checkpoints)."""
    runs = [r for r in phase14.all_runs() if r.name.startswith("full_S0_N2K1")]
    values: List[Tuple[int, Optional[float]]] = sorted((r.seed, read_result(save_dir, r, "best", protocol)) for r in runs)
    done = [v for _, v in values if v is not None]
    spread = (f"mean {statistics.mean(done):.2f}, std {statistics.stdev(done):.2f} over {len(done)} seeds"
              if len(done) >= 2 else "needs at least two finished seeds")
    cells = ", ".join(f"seed {s}: {fmt(v)}" for s, v in values)
    return f"### Seed spread - full model, S0 2-way 1-shot (best)\n\n{cells}; {spread}. Paper S0: 88.53."


def report(save_dir: str, protocol: str) -> str:
    parts = [f"## Phase-14 results ({protocol}, mIoU %)", "",
             "`best` = best validation checkpoint, `last` = last epoch (D-15); Avg = mean of S0 and S1."]
    for table in ("T4", "T5", "T2"):
        parts += ["", table_markdown(save_dir, protocol, table)]
    parts += ["", seed_markdown(save_dir, protocol)]
    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--save_dir", default="log_phase14")
    p.add_argument("--protocol", default="fixed100", choices=["fixed100", "random600"])
    args = p.parse_args(argv)
    print(report(os.path.abspath(args.save_dir), args.protocol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
