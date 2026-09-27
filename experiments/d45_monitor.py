"""D-45: M2 (VICReg on the query features) - the training monitor, the collapse census and the rules.

  census   participation ratio of the unit query features and U's mIoU on the first 300 valid episodes, for every
           kept S1 checkpoint (inference, next to the training)
  watch    during training: at every validation epoch of each arm, the same two numbers on its current weights;
           an arm below ratio 8 at epoch 24 is stopped (its process receives SIGTERM)
  decide   rules D45.1-D45.5 from part B's full measurement, the test draws and the monitor

    python experiments/d45_monitor.py census --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint cr:ours:1:<pt> ...
    python experiments/d45_monitor.py watch --data_path datasets/S3DIS/blocks_bs1_s1 --run m2a:<run dir>:<pid> ...
    python experiments/d45_monitor.py decide
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import d39_eval as d39  # noqa: E402
from experiments import p0_em_probe as p0  # noqa: E402
from experiments import p5_condition_probe as p5  # noqa: E402
from experiments import p6_prototype_probe as p6  # noqa: E402
from experiments import p9_placement_probe as p9  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_d45"
MONITOR_EPISODES = 300  # the first valid episodes [DECISION D-45]
VALID_EVERY = 4  # CR's schedule: a validation every 4 epochs
STOP_EPOCH, STOP_RATIO = 24, 8.0  # early stop: ratio below 8 at epoch 24 [DECISION D-45]
MECHANISM_RATIO = 12.0  # D45.1 (c)
BASE_GAIN = 1.0  # D45.2
ARMS = ("m2a", "m2b")


# ------------------------------------------------------------------ pure

class Moments:
    """Running first and second moments of row vectors (float64) -> covariance and participation ratio."""

    def __init__(self):
        self.s1, self.s2, self.n = None, None, 0

    def add(self, x: torch.Tensor) -> None:
        x = x.double()  # [n, D]
        self.s1 = x.sum(0) if self.s1 is None else self.s1 + x.sum(0)
        self.s2 = x.T @ x if self.s2 is None else self.s2 + x.T @ x
        self.n += x.shape[0]

    def cov(self) -> torch.Tensor:
        mean = self.s1 / self.n  # [D]
        return self.s2 / self.n - torch.outer(mean, mean)  # [D, D]

    def ratio(self) -> float:
        return p9.participation_ratio(self.cov())


def early_stop(epoch: int, ratio: float) -> bool:
    """[DECISION D-45]: the first measurement at or after epoch 24 decides."""
    return epoch >= STOP_EPOCH and ratio < STOP_RATIO


def decide(parts: Dict[str, Optional[Dict]], draws: Dict, monitor: List[Dict]) -> List[Tuple[str, str]]:
    """Rules D45.1-D45.4 [DECISION D-45]. parts: part B's result per arm (None if the arm was stopped)."""
    v = []
    mech = {}
    for arm in ARMS:
        part = parts.get(arm)
        stopped = [r for r in monitor if r.get("arm") == arm and r.get("stopped")]
        if part is None:
            v.append((f"D45.1 {arm} not measured", "stopped early" if stopped else "no part B result"))
            mech[arm] = False
            continue
        pr = part["collapse"]["participation_ratio"]
        mech[arm] = pr >= MECHANISM_RATIO
        v.append((f"D45.1 {arm} mechanism {'holds' if mech[arm] else 'fails'}",
                  f"participation ratio {pr:.2f} (limit {MECHANISM_RATIO:g}); base-span share "
                  f"{part['collapse']['span_share_median']:.3f}; missed purity {part['purity']['miss']:.3f}"))
    if not any(mech.values()):
        v.append(("D45.4 stop: VICReg on the features does not undo the collapse at these weights", ""))
        return v
    missing = [d for d in d39.DRAWS if d not in draws]
    if missing:
        v.append(("incomplete", f"missing draws {missing}"))
        return v
    base = []
    for arm in ARMS:
        if not mech[arm]:
            continue
        p_fx, rand = d39.paired(draws, "cr:U", f"{arm}:U")
        ok = d39.holds(p_fx, rand, BASE_GAIN)
        v.append((f"D45.2 {arm} U {'holds' if ok else 'fails'} at +{BASE_GAIN:g}",
                  f"{arm} - cr fixed100 {p_fx['gain']:+.2f} [{p_fx['ci_low']:+.2f}, {p_fx['ci_high']:+.2f}]; "
                  f"random600 {[round(g, 2) for g in rand]}"))
        if ok:
            base.append(arm)
    if base:
        best = max(base, key=lambda a: parts[a]["miou"]["U"])
        v.append(("D45.2 base", f"{best} (valid U {100 * parts[best]['miou']['U']:.2f})"))
    else:
        v.append(("D45.3 the space widened, U does not use it: part B's preconditions decide the next block", ""))
    return v


# ------------------------------------------------------------------ GPU

@torch.no_grad()
def measure(rule, data_path: str, device, n_episodes: int) -> Dict[str, float]:
    """Participation ratio of the unit query features and U's mIoU (and the model's) on the first valid episodes."""
    from pipeline.episodes import make_episode, read_class_names

    names = read_class_names(data_path, "s3dis")
    items, test_classes = p6.episodes("valid", data_path, n_episodes)
    mom = Moments()
    counts = {"U": [], "model": []}
    for item in items:
        e = make_episode(item, names).to(device)
        f_q, m_eff, _, logits = rule(e)
        p0.check_identity(f_q, m_eff, logits)
        f_s = p5.support_features(rule, e)
        mom.add(F.normalize(f_q, dim=-1).reshape(-1, f_q.shape[-1]))  # [B_q·P, D], blocks and points only
        gt = e.query_y.cpu().numpy()
        preds = {"U": p6.rule_logits(f_q, p6.base_rows(f_q, f_s, e.support_y)).argmax(-1), "model": logits.argmax(-1)}
        for k, pr in preds.items():
            counts[k].append(p0.episode_counts(pr.cpu().numpy(), gt, e.sampled_classes, test_classes))
    out = {k: float(p0.miou_from_counts(np.stack(v).sum(0))) for k, v in counts.items()}
    out.update(ratio=mom.ratio(), episodes=len(counts["U"]))
    return out


def load(spec: str, device):
    ck = p0.parse_checkpoint(spec)
    rule, protocol, config = r2.load(ck, 1, device)  # eval.py's protocol guard [DECISION D-22]
    return ck, rule


def snapshot(run_dir: str, out_path: str) -> Optional[int]:
    """Current weights of a running arm as an evaluable checkpoint: resume.pt's model with best.pt's configuration.
    Both files are our own and written atomically by train.py; returns the epoch, or None before the first
    validation."""
    best, resume = os.path.join(run_dir, "best.pt"), os.path.join(run_dir, "resume.pt")
    if not (os.path.isfile(best) and os.path.isfile(resume)):
        return None
    b = torch.load(best, map_location="cpu", weights_only=False)  # our checkpoint (numpy states in args)
    r = torch.load(resume, map_location="cpu", weights_only=False)  # our checkpoint (optimiser, RNG states)
    epoch = int(r["state"]["epoch"])  # train.py's resume state [train.py: atomic_save of resume.pt]
    torch.save({"model": r["model"], "config": r["config"], "epoch": epoch, "args": b["args"]}, out_path)
    return epoch


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def watch(runs: List[Tuple[str, str, int]], data_path: str, device, n_episodes: int, out_dir: str,
          poll: float) -> None:
    log = os.path.join(out_dir, "monitor.jsonl")
    target = {name: VALID_EVERY for name, _, _ in runs}
    decided = {name: False for name, _, _ in runs}
    while True:
        running = False
        for name, run_dir, pid in runs:
            is_alive = alive(pid)
            running = running or is_alive
            snap = os.path.join(out_dir, f"monitor_{name}.pt")
            epoch = snapshot(run_dir, snap)
            if epoch is None or epoch < target[name]:
                continue
            _, rule = load(f"{name}:ours:1:{snap}", device)
            res = measure(rule, data_path, device, n_episodes)
            del rule
            torch.cuda.empty_cache()
            rec = {"arm": name, "epoch": epoch, **res, "time": time.strftime("%H:%M:%S")}
            target[name] = (epoch // VALID_EVERY + 1) * VALID_EVERY
            if not decided[name] and epoch >= STOP_EPOCH:
                decided[name] = True
                if early_stop(epoch, res["ratio"]) and is_alive:
                    os.kill(pid, signal.SIGTERM)
                    rec["stopped"] = True
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[watch] {name} epoch {epoch} ratio {res['ratio']:.2f} U {100 * res['U']:.2f} "
                  f"model {100 * res['model']:.2f}" + (" STOPPED" if rec.get("stopped") else ""), flush=True)
        if not running:
            return
        time.sleep(poll)


# ------------------------------------------------------------------ CLI

def read_json(path: str) -> Optional[Dict]:
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["census", "watch", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--run", action="append", default=[], help="name:run_dir:pid")
    p.add_argument("--episodes", type=int, default=MONITOR_EPISODES)
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--tag", default="")
    p.add_argument("--out_dir", default=OUT_DIR)
    args = p.parse_args(argv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.stage == "decide":
        parts = {a: read_json(os.path.join(out_dir, f"modules_cr_valid_{a}{args.tag}.json")) for a in ARMS}
        draws = {}
        for d in d39.DRAWS:
            path = os.path.join(out_dir, p6.stem(d, args.tag) + ".json")
            if os.path.isfile(path):
                draws[d] = (read_json(path), dict(np.load(path.replace(".json", "_counts.npz"))))
        mon_path = os.path.join(out_dir, "monitor.jsonl")
        monitor = [json.loads(line) for line in open(mon_path)] if os.path.isfile(mon_path) else []
        for name, text in decide(parts, draws, monitor):
            print(f"{name:70s} {text}")
        return 0
    if not args.data_path:
        p.error("--data_path is required")
    device = torch.device("cuda")
    if args.stage == "census":
        res = {}
        for spec in args.checkpoint:  # report only: a checkpoint that cannot be read is recorded, not fatal
            name = spec.split(":", 1)[0]
            try:
                ck, rule = load(spec, device)
                res[name] = {**measure(rule, args.data_path, device, args.episodes), "checkpoint": vars(ck)}
                del rule
            except Exception as err:  # noqa: BLE001
                res[name] = {"error": f"{type(err).__name__}: {err}"}
                print(f"[census] {name}: ERROR {res[name]['error']}", flush=True)
                continue
            finally:
                torch.cuda.empty_cache()
            print(f"[census] {name}: ratio {res[name]['ratio']:.2f} U {100 * res[name]['U']:.2f} "
                  f"model {100 * res[name]['model']:.2f}", flush=True)
        p6.save(res, None, f"census{args.tag}", out_dir)
        return 0
    runs = []
    for r in args.run:
        name, run_dir, pid = r.rsplit(":", 2)
        runs.append((name, run_dir, int(pid)))
    watch(runs, args.data_path, device, args.episodes, out_dir, args.poll)
    return 0


if __name__ == "__main__":
    sys.exit(main())
