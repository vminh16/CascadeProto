"""Phase-14 run queue: full S3DIS training runs for Tables 2, 4 and 5 of the paper, then evaluation.

    python experiments/phase14.py --data_path datasets/S3DIS/blocks_bs1_s1 --priority P1          # run P1
    python experiments/phase14.py --data_path datasets/S3DIS/blocks_bs1_s1 --priority P1 P2 --list  # show plan

Every step is idempotent: training resumes from `resume.pt` and is skipped once `last.pt` exists; an
evaluation is skipped once its JSON result exists. Rerun the same command after an interruption.
Results live next to the checkpoints (`eval_<best|last>_<protocol>.json`); experiments/summarize.py
turns them into tables.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import train  # noqa: E402  (run_dir and argument parsing are the single source of run names)

PRIORITIES = ("P1", "P2", "P2s", "P3", "P4")

# Table 4 rows as switches [DECISION D-17]; defaults of train.py are the full model.
ROWS: Dict[str, Dict[str, str]] = {
    "baseline": {"use_lma": "false", "num_stages": "0"},
    "lma": {"use_lma": "true", "num_stages": "0"},
    "gate": {"use_lma": "true", "num_stages": "1"},  # = Table 5, T = 1
    "cascade": {"use_lma": "true", "num_stages": "4", "use_adrm": "false"},
    "full": {},  # = Table 5, T = 4
}


@dataclass(frozen=True)
class Run:
    name: str
    priority: str
    cvfold: int
    n_way: int
    k_shot: int
    flags: Tuple[Tuple[str, str], ...]
    seed: int = 0
    tables: Tuple[Tuple[str, str], ...] = ()  # (table, row) cells this run fills
    random600: bool = False  # also evaluate with the paper's wording of the protocol (D-08 flag)


def _run(priority, row, fold, n=2, k=1, seed=0, flags=None, tables=(), random600=False) -> Run:
    flags = dict(ROWS[row] if flags is None else flags)
    name = f"{row}_S{fold}_N{n}K{k}" + (f"_seed{seed}" if seed else "")
    return Run(name, priority, fold, n, k, tuple(sorted(flags.items())), seed, tuple(tables), random600)


def all_runs() -> List[Run]:
    runs = [
        _run("P1", "full", 0, tables=[("T4", "full"), ("T5", "T4"), ("T2", "N2K1")], random600=True),
        _run("P1", "baseline", 0, tables=[("T4", "baseline")], random600=True),
        _run("P2", "baseline", 1, tables=[("T4", "baseline")]),
        _run("P2", "full", 1, tables=[("T4", "full"), ("T5", "T4"), ("T2", "N2K1")]),
    ]
    for row in ("lma", "gate", "cascade"):
        tables = [("T4", row)] + ([("T5", "T1")] if row == "gate" else [])
        runs += [_run("P2", row, fold, tables=tables) for fold in (0, 1)]
    runs += [_run("P2s", "full", 0, seed=s) for s in (1, 2)]
    for n, k in ((2, 5), (3, 1), (3, 5)):
        runs += [_run("P3", "full", fold, n, k, tables=[("T2", f"N{n}K{k}")]) for fold in (0, 1)]
    for t in (2, 3, 5, 6):
        runs += [_run("P4", f"T{t}", fold, flags={"num_stages": str(t)}, tables=[("T5", f"T{t}")])
                 for fold in (0, 1)]
    return runs


def train_argv(run: Run, data_path: str, save_dir: str) -> List[str]:
    argv = ["--dataset", "s3dis", "--data_path", data_path, "--cvfold", str(run.cvfold),
            "--n_way", str(run.n_way), "--k_shot", str(run.k_shot), "--seed", str(run.seed),
            "--save_dir", save_dir, "--resume", "true"]
    for key, value in run.flags:
        argv += [f"--{key}", value]
    return argv


def run_dir(run: Run, save_dir: str) -> str:
    return train.run_dir(train.parse_args(train_argv(run, "unused", save_dir)))


def eval_argv(run: Run, data_path: str, checkpoint: str, protocol: str, result_json: str) -> List[str]:
    return ["--dataset", "s3dis", "--data_path", data_path, "--cvfold", str(run.cvfold),
            "--n_way", str(run.n_way), "--k_shot", str(run.k_shot), "--checkpoint", checkpoint,
            "--eval_protocol", protocol, "--result_json", result_json]


def result_path(run: Run, save_dir: str, which: str, protocol: str) -> str:
    return os.path.join(run_dir(run, save_dir), f"eval_{which}_{protocol}.json")


def pending_steps(run: Run, data_path: str, save_dir: str, python: str = sys.executable) -> List[Tuple[str, List[str]]]:
    """(label, command) for every step of `run` that has not produced its output yet, in order."""
    out = run_dir(run, save_dir)
    steps = []
    if not os.path.isfile(os.path.join(out, "last.pt")):
        steps.append((f"{run.name}: train", [python, os.path.join(REPO, "train.py")] + train_argv(run, data_path, save_dir)))
    protocols = ["fixed100"] + (["random600"] if run.random600 else [])
    for which in ("best", "last"):
        for protocol in protocols:
            target = result_path(run, save_dir, which, protocol)
            if not os.path.isfile(target):
                checkpoint = os.path.join(out, f"{which}.pt")
                steps.append((f"{run.name}: eval {which} {protocol}",
                              [python, os.path.join(REPO, "eval.py")] + eval_argv(run, data_path, checkpoint, protocol, target)))
    return steps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_path", required=True)
    p.add_argument("--save_dir", default="log_phase14")
    p.add_argument("--priority", nargs="+", default=["P1"], choices=PRIORITIES)
    p.add_argument("--list", action="store_true", help="print the pending steps and exit")
    args = p.parse_args(argv)
    # absolute paths: the checks here and the subprocesses (cwd = REPO) must see the same files
    args.data_path, args.save_dir = os.path.abspath(args.data_path), os.path.abspath(args.save_dir)
    runs = [r for r in all_runs() if r.priority in args.priority]
    for run in runs:
        for label, command in pending_steps(run, args.data_path, args.save_dir):
            print(f"[phase14] {label}\n  {' '.join(command)}", flush=True)
            if not args.list:
                subprocess.run(command, check=True, cwd=REPO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
