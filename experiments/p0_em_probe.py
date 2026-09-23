"""P0 of [DECISION D-26]: query-side, entropy-weighted EM refinement on trained checkpoints, no training.

    python experiments/p0_em_probe.py select --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint vipseg_S1:vipseg:1:vipseg_S1_N2_K1.pt --checkpoint ours_S1:ours:1:log_s1/.../last.pt
    python experiments/p0_em_probe.py test --data_path ... --checkpoint ...              # frozen setting
    python experiments/p0_em_probe.py decide                                             # rules P0.1-P0.6

Question: on features and prototypes that a model already learned, does re-estimating each prototype
from the query's own confident points (models/transductive.py) raise the mIoU, and does the entropy
weight single out the right points? No parameter is trained, so the comparison of an arm with the
model's own prediction is **paired on identical episodes and weights**: seed noise of training does not
enter it, and the paired bootstrap over episodes gives its uncertainty.

Protocol, fixed before any run [DECISION D-26]:
* `select` scores the grid on the S1 **valid** draw of S1 checkpoints only [DECISION D-15] [DECISION D-22]
  and freezes the setting with the largest mean gain over them (oracle arms never selected).
* `test` scores that frozen setting once on the fixed100 **test** draw: S1 checkpoints on S1, S0
  checkpoints on S0 (held out until now). Arms: the model, the frozen setting, the same setting with
  `weight=none` (the entropy ablation) and the oracle (query labels, an upper bound, never a result).
* `decide` applies the rules of D-26 to the test JSONs.
Every checkpoint passes eval.py's protocol guard: it is scored only on its own fold's test classes.
"""

import argparse
import glob
import itertools
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models.transductive import em_step, entropy_diagnostics, responsibilities  # noqa: E402
from pipeline.metrics_alt import VIPSEG_EPS  # noqa: E402

OUT_DIR = "results/phase16_p0"
# Grid of D-26. ||u_c|| is at most the confident mass fraction pi_c of class c (models/transductive.py),
# so the query's pull against the unit prior is at most kappa * pi_c: the note's {0.25, 0.5, 1}
# [research note 2026-09-23 §7.1] would move a foreground class with pi_c ~ 0.1-0.3 (an estimate, not
# measured; `confident_mass` below records it) by a few percent only, hence kappa up to 16.
# T: 1-3, soft k-means gains saturate after one step [Ren §3.1.1].
# Weights: entropy (D-26), none (its ablation), SSP's thresholds [SSP §3.2].
KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MAX_STEPS = 3
WEIGHT_ARMS = ("entropy", "none", "ssp")
ORACLE_REPLACE = 1e4  # kappa large enough that the direction is the query's own class mean
BOOTSTRAP = 2000
IDENTITY_RTOL = 1e-4  # |L_model - F M^T| <= rtol * max|L_model|, float32 on the GPU


# ------------------------------------------------------------------ the model's own scoring rule

class VIPSegScoringRule:
    """F^q and the effective prototypes of VIP-Seg's released model, read through forward hooks.

    VIP-Seg scores `L = sum_t w_t F^q (M^t)^T` [VIPSEG models/vipseg.py:152-174], which is `F^q M_eff^T`
    with `M_eff = sum_t w_t M^t`. The hooks record `F^q` (the second call of `fc`, after the support),
    the prototype after every step (PDM steps add the outer residual, vipseg.py:157) and the gating
    weights. The inherited model is never edited (AGENTS guardrail 2); the returned logits are checked
    against the model's own on every episode.
    """

    def __init__(self, baseline):
        self.baseline = baseline
        self.model = baseline.model
        self.fc_out: List[torch.Tensor] = []
        self.step_in: List[torch.Tensor] = []
        self.step_out: List[torch.Tensor] = []
        self.gate: List[torch.Tensor] = []
        m = self.model
        self.handles = [m.fc.register_forward_hook(lambda mod, i, o: self.fc_out.append(o)),
                        m.gating_network.register_forward_hook(lambda mod, i, o: self.gate.append(o))]
        for module in m.vip_module:
            self.handles.append(module.register_forward_pre_hook(lambda mod, i: self.step_in.append(i[2])))
            self.handles.append(module.register_forward_hook(lambda mod, i, o: self.step_out.append(o)))

    def __call__(self, episode) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.fc_out.clear(), self.step_in.clear(), self.step_out.clear(), self.gate.clear()
        logits = self.baseline(episode).logits  # [B_q, P, N+1]
        if len(self.fc_out) != 2 or len(self.gate) != 1 or len(self.step_out) != len(self.model.vip_module):
            raise RuntimeError(f"unexpected VIP-Seg call pattern: fc {len(self.fc_out)}, gate {len(self.gate)}, "
                               f"steps {len(self.step_out)}")
        f_q = self.fc_out[1].permute(0, 2, 1)  # [B_q, 128, P] -> [B_q, P, 128]
        stages = [out + inp if t % 2 == 1 else out  # [B_q, N+1, D], outer residual on PDM steps
                  for t, (inp, out) in enumerate(zip(self.step_in, self.step_out))]
        m_eff = torch.einsum("bt,tbcd->bcd", self.gate[0], torch.stack(stages))  # [B_q, N+1, D]
        return f_q, m_eff, logits

    def close(self):
        for h in self.handles:
            h.remove()


class CascadeProtoScoringRule:
    """F^q and P for our checkpoints without stages or LMA: `L = F^q P^T` (Eq.23 on Eq.3's prototypes).

    The P0 plan names our baseline (`num_stages=0`); other configurations raise rather than being
    approximated. The logits are checked against the model's own forward.
    """

    def __init__(self, model):
        cfg = model.config
        if cfg.num_stages != 0 or cfg.use_lma:
            raise NotImplementedError(f"P0 reads the scoring rule of num_stages=0 without LMA; got "
                                      f"num_stages={cfg.num_stages}, use_lma={cfg.use_lma} [DECISION D-26]")
        self.model = model

    def __call__(self, episode) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch.nn.functional as F

        from models.prototypes import point_prototypes

        cfg = self.model.config
        logits = self.model(episode).logits  # [B_q, P, N+1]
        f_s, f_q = self.model.features.encode_episode(episode.support_x, episode.query_x)  # [N,K,P,D], [B_q,P,D]
        p = point_prototypes(f_s, episode.support_y)  # [N+1, D] (Eq.3)
        if cfg.l2norm_point_proto:
            p = F.normalize(p, dim=-1)  # [N+1, D]
        if cfg.logit_scale == "sqrt_D":
            p = p / f_q.shape[-1] ** 0.5  # [N+1, D], folds the scale into the prototype
        return f_q, p.unsqueeze(0).expand(f_q.shape[0], -1, -1), logits  # [B_q, P, D], [B_q, N+1, D], [B_q,P,N+1]

    def close(self):
        pass


def check_identity(f_q: torch.Tensor, prior: torch.Tensor, logits: torch.Tensor) -> None:
    """The extracted rule must reproduce the model's logits; a mismatch means the probe reads the wrong tensors."""
    rebuilt = torch.einsum("bpd,bcd->bpc", f_q, prior)  # [B_q, P, N+1]
    err = (rebuilt - logits).abs().max().item()
    if err > IDENTITY_RTOL * max(logits.abs().max().item(), 1.0):
        raise RuntimeError(f"scoring rule does not reproduce the model's logits: max |diff| = {err:.3e}")


# ------------------------------------------------------------------ arms and counts

def grid_arms() -> List[Dict]:
    """Every (weight, kappa, steps) of the selection grid, oracle arms included (never selected)."""
    arms = [dict(weight=w, kappa=k, steps=t)
            for w, k in itertools.product(WEIGHT_ARMS, KAPPAS) for t in range(1, MAX_STEPS + 1)]
    arms += [dict(weight="oracle", kappa=k, steps=1) for k in KAPPAS]
    arms.append(dict(weight="oracle", kappa=ORACLE_REPLACE, steps=1))
    return arms


def arm_name(arm: Dict) -> str:
    return f"{arm['weight']}_k{arm['kappa']:g}_T{arm['steps']}"


def run_chains(f_q, prior, logits0, arms: List[Dict], labels) -> Dict[str, torch.Tensor]:
    """Predictions [B_q, P] of every arm; arms that share (weight, kappa) share their EM chain."""
    preds = {"model": logits0.argmax(dim=-1)}  # [B_q, P]
    chains = sorted({(a["weight"], a["kappa"]) for a in arms})
    wanted = {arm_name(a) for a in arms}
    for weight, kappa in chains:
        logits = logits0  # [B_q, P, N+1]
        for t in range(1, MAX_STEPS + 1):
            _, logits = em_step(f_q, prior, logits, kappa, weight, labels)  # [B_q, P, N+1]
            name = arm_name(dict(weight=weight, kappa=kappa, steps=t))
            if name in wanted:
                preds[name] = logits.argmax(dim=-1)  # [B_q, P]
            if not any(arm_name(dict(weight=weight, kappa=kappa, steps=s)) in wanted
                       for s in range(t + 1, MAX_STEPS + 1)):
                break
    return preds


def episode_counts(pred: np.ndarray, gt: np.ndarray, label2class, test_classes: List[int]) -> np.ndarray:
    """[3, C+1]: GT count, predicted count and TP per global class, as `accumulated_counts` for one episode."""
    n = len(test_classes) + 1
    lut = np.array([0] + [test_classes.index(int(c)) + 1 for c in label2class])  # [N+1]
    g, p = lut[gt.ravel()], lut[pred.ravel()]  # [B_q * P]
    return np.stack([np.bincount(g, minlength=n), np.bincount(p, minlength=n),
                     np.bincount(g[gt.ravel() == pred.ravel()], minlength=n)]).astype(np.float64)


def miou_from_counts(c: np.ndarray) -> np.ndarray:
    """VIP-Seg's accumulated mIoU (background excluded) from summed counts [..., 3, C+1] -> [...]."""
    gt, pr, tp = c[..., 0, :], c[..., 1, :], c[..., 2, :]
    return (tp / (gt + pr - tp + VIPSEG_EPS))[..., 1:].mean(axis=-1)


def class_ious(c: np.ndarray) -> np.ndarray:
    gt, pr, tp = c[0], c[1], c[2]
    return tp / (gt + pr - tp + VIPSEG_EPS)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Gain b - a in mIoU points, resampling episodes jointly; a, b [E, 3, C+1] per-episode counts."""
    rng = np.random.default_rng(seed)
    e = a.shape[0]
    w = rng.multinomial(e, np.full(e, 1.0 / e), size=BOOTSTRAP).astype(np.float64)  # [B, E]
    ga = miou_from_counts(np.einsum("be,exy->bxy", w, a))  # [B]
    gb = miou_from_counts(np.einsum("be,exy->bxy", w, b))  # [B]
    d = 100.0 * (gb - ga)  # [B]
    return {"gain": 100.0 * float(miou_from_counts(b.sum(0)) - miou_from_counts(a.sum(0))),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "p_le_zero": float(np.mean(d <= 0))}


# ------------------------------------------------------------------ one checkpoint

def parse_checkpoint(spec: str) -> SimpleNamespace:
    """name:kind:trained_fold:path, kind in {vipseg, ours}."""
    name, kind, fold, path = spec.split(":", 3)
    if kind not in ("vipseg", "ours"):
        raise ValueError(f"checkpoint kind must be vipseg or ours, got {kind!r}")
    return SimpleNamespace(name=name, kind=kind, fold=int(fold), path=path)


def load_rule(ck: SimpleNamespace, cvfold: int, device):
    """The model behind eval.py's loader and protocol guard [DECISION D-22], wrapped as a scoring rule."""
    import eval as eval_cli

    args = SimpleNamespace(checkpoint=ck.path, model="vipseg" if ck.kind == "vipseg" else "cascadeproto",
                           cvfold=cvfold, checkpoint_cvfold=ck.fold, allow_seen_classes=False)
    model = eval_cli.load_model(args, device)
    protocol = eval_cli.protocol_check(args, model.checkpoint_args)  # raises on seen classes
    model.eval()
    rule = VIPSegScoringRule(model) if ck.kind == "vipseg" else CascadeProtoScoringRule(model)
    return rule, protocol


@torch.no_grad()
def score_checkpoint(ck, data_path: str, cvfold: int, mode: str, arms: List[Dict], device,
                     max_episodes=None) -> Dict:
    from torch.utils.data import DataLoader, Subset

    from pipeline.episodes import EpisodeCollate, build_eval_dataset, read_class_names
    from pipeline.evaluation import accumulated_miou

    rule, protocol = load_rule(ck, cvfold, device)
    names = read_class_names(data_path, "s3dis")
    dataset = build_eval_dataset(data_path, "s3dis", cvfold, 2, 1, mode=mode, seed=0)
    test_classes = [int(c) for c in np.asarray(dataset.classes)]
    view = dataset if max_episodes is None else Subset(dataset, range(min(max_episodes, len(dataset))))
    loader = DataLoader(view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))
    counts: Dict[str, List[np.ndarray]] = {}
    base_preds, gts, l2c = [], [], []
    diag = {"count": 0, "correct": 0, "entropy_sum": 0.0}
    mass = {"bg": [], "fg": []}
    for (episode,) in loader:
        episode = episode.to(device)
        f_q, prior, logits0 = rule(episode)  # [B_q,P,D], [B_q,N+1,D], [B_q,P,N+1]
        check_identity(f_q, prior, logits0)
        labels = episode.query_y  # [B_q, P], used by the oracle arms and the diagnostics only
        preds = run_chains(f_q, prior, logits0, arms, labels)
        gt = labels.cpu().numpy()
        for name, pred in preds.items():
            counts.setdefault(name, []).append(episode_counts(pred.cpu().numpy(), gt, episode.sampled_classes,
                                                              test_classes))
        base_preds.append(preds["model"].cpu().numpy()), gts.append(gt), l2c.append(episode.sampled_classes)
        d = entropy_diagnostics(logits0, labels)
        diag["count"] += d["count"].cpu().numpy()
        diag["correct"] += d["correct"].cpu().numpy()
        diag["entropy_sum"] += float(d["entropy_sum"])
        wr = responsibilities(logits0, "entropy")  # [B_q, P, N+1]
        frac = (wr.sum(dim=1) / wr.shape[1]).cpu().numpy()  # [B_q, N+1], confident mass fraction
        mass["bg"].append(float(frac[:, 0].mean())), mass["fg"].append(float(frac[:, 1:].mean()))
    rule.close()

    class _Quiet:
        def cprint(self, text):
            pass

    stacked = {k: np.stack(v) for k, v in counts.items()}  # name -> [E, 3, C+1]
    primary = accumulated_miou(_Quiet(), base_preds, gts, l2c, test_classes)  # VIP-Seg's metric [D-08]
    if abs(primary - miou_from_counts(stacked["model"].sum(0))) > 1e-9:
        raise RuntimeError("count-based mIoU differs from VIP-Seg's evaluate_metric")
    n_pts = float(diag["count"].sum())
    return {
        "checkpoint": vars(ck), "cvfold": cvfold, "mode": mode, "protocol": protocol,
        "episodes": len(view), "test_classes": test_classes,
        "miou": {k: float(miou_from_counts(v.sum(0))) for k, v in stacked.items()},
        "class_iou": {k: class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
        "entropy": {"mean_normalized_entropy": diag["entropy_sum"] / n_pts,
                    "bins": ["w<0.5", "0.5<=w<0.9", "w>=0.9"],
                    "fraction": (diag["count"] / n_pts).tolist(),
                    "accuracy": (diag["correct"] / np.maximum(diag["count"], 1)).tolist()},
        "confident_mass": {"bg": float(np.mean(mass["bg"])), "fg": float(np.mean(mass["fg"]))},
    }, stacked


def save(result: Dict, stacked: Dict[str, np.ndarray], stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, stem + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, stem + "_counts.npz"), **stacked)


# ------------------------------------------------------------------ stages

def select_setting(results: List[Dict]) -> Dict:
    """The non-oracle arm with the largest mean gain over the selection checkpoints."""
    arms = [a for a in grid_arms() if a["weight"] != "oracle"]
    gains = {arm_name(a): float(np.mean([100.0 * (r["miou"][arm_name(a)] - r["miou"]["model"]) for r in results]))
             for a in arms}
    best = max(arms, key=lambda a: gains[arm_name(a)])
    return {"frozen": best, "name": arm_name(best), "mean_valid_gain": gains[arm_name(best)],
            "all_gains": dict(sorted(gains.items(), key=lambda kv: -kv[1]))}


def test_arms(frozen: Dict) -> List[Dict]:
    return [frozen, dict(frozen, weight="none"), dict(frozen, weight="oracle", steps=1),
            dict(weight="oracle", kappa=ORACLE_REPLACE, steps=1)]


def cmd_select(args, device) -> int:
    results = []
    for spec in args.checkpoint:
        ck = parse_checkpoint(spec)
        if ck.fold != 1:
            raise ValueError(f"{ck.name}: selection uses S1 checkpoints only, S0 stays held out [DECISION D-22]")
        result, stacked = score_checkpoint(ck, args.data_path, 1, "valid", grid_arms(), device, args.max_episodes)
        save(result, stacked, f"select_{ck.name}")
        results.append(result)
        print(f"[select] {ck.name}: model {result['miou']['model']:.4f} | entropy bins "
              f"{result['entropy']['fraction']} acc {result['entropy']['accuracy']}", flush=True)
    choice = select_setting(results)
    choice["checkpoints"] = [r["checkpoint"] for r in results]
    with open(os.path.join(OUT_DIR, "selection.json"), "w") as f:
        json.dump(choice, f, indent=1)
    print(f"[select] frozen: {choice['name']} (mean valid gain {choice['mean_valid_gain']:+.2f})", flush=True)
    return 0


def cmd_test(args, device) -> int:
    with open(os.path.join(OUT_DIR, "selection.json")) as f:
        frozen = json.load(f)["frozen"]
    arms = test_arms(frozen)
    names = [arm_name(a) for a in arms]
    for spec in args.checkpoint:
        ck = parse_checkpoint(spec)
        result, stacked = score_checkpoint(ck, args.data_path, ck.fold, "test", arms, device, args.max_episodes)
        result["frozen"] = names[0]
        result["paired"] = {"frozen_vs_model": paired_bootstrap(stacked["model"], stacked[names[0]]),
                            "frozen_vs_unweighted": paired_bootstrap(stacked[names[1]], stacked[names[0]]),
                            "oracle_same_kappa_vs_model": paired_bootstrap(stacked["model"], stacked[names[2]]),
                            "oracle_replace_vs_model": paired_bootstrap(stacked["model"], stacked[names[3]])}
        save(result, stacked, f"test_{ck.name}")
        p = result["paired"]["frozen_vs_model"]
        print(f"[test] {ck.name} S{ck.fold}: model {result['miou']['model']:.4f} -> {result['miou'][names[0]]:.4f} "
              f"({p['gain']:+.2f}, 95% CI {p['ci_low']:+.2f}..{p['ci_high']:+.2f})", flush=True)
    return 0


def decide(tests: List[Dict]) -> List[Tuple[str, str]]:
    """Rules P0.1-P0.6 of [DECISION D-26], applied to the test JSONs."""
    s1 = [t for t in tests if t["cvfold"] == 1]
    s0 = [t for t in tests if t["cvfold"] == 0]
    g = {t["checkpoint"]["name"]: t["paired"]["frozen_vs_model"]["gain"] for t in tests}
    verdicts = []
    if len(s1) < 2 or len(s0) < 2:
        return [("incomplete", f"need two S1 and two S0 test results, have {len(s1)} and {len(s0)}")]
    s1g = [t["paired"]["frozen_vs_model"]["gain"] for t in s1]
    s0g = [t["paired"]["frozen_vs_model"]["gain"] for t in s0]
    if min(s1g) >= 1.5 and min(s0g) >= 1.0:
        verdicts.append(("P0.1 go", f"S1 gains {s1g}, S0 gains {s0g}: train R2"))
    elif max(s1g) < 0.5:
        verdicts.append(("P0.2 stop", f"S1 gains {s1g} all below +0.5: drop the direction"))
    else:
        verdicts.append(("P0.3 in between",
                         f"S1 {s1g}, S0 {s0g}: train R2 only with learned kappa, expect the low end"))
    for t in s0:
        ci = t["paired"]["frozen_vs_model"]
        verdicts.append(("P0.4 held-out gain " + t["checkpoint"]["name"],
                         f"{ci['gain']:+.2f} [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]: "
                         + ("above 0" if ci["ci_low"] > 0 else "below 0" if ci["ci_high"] < 0 else "contains 0")))
    ew = [t["paired"]["frozen_vs_unweighted"] for t in s0]
    if not s0[0]["frozen"].startswith("entropy_"):
        verdicts.append(("P0.5 entropy weight", f"not claimable: selection froze {s0[0]['frozen']}"))
    else:
        verdicts.append(("P0.5 entropy weight", "claimable" if all(e["ci_low"] > 0 for e in ew) else
                         f"not claimable (frozen - unweighted on S0: {[round(e['gain'], 2) for e in ew]})"))
    for t in tests:
        if t["checkpoint"]["kind"] == "vipseg":
            o = t["paired"]["oracle_replace_vs_model"]["gain"]
            bound = " < 10: the features, not the prototypes, bound this model" if o < 10 else ""
            verdicts.append((f"P0.6 oracle {t['checkpoint']['name']}", f"+{o:.2f}{bound}"))
        frozen = t["frozen"]
        drop = [round(100 * (b - a), 2) for a, b in zip(t["class_iou"]["model"][1:], t["class_iou"][frozen][1:])]
        if g[t["checkpoint"]["name"]] > 0 and min(drop) < -3.0:
            verdicts.append((f"collapse watch {t['checkpoint']['name']}", f"per-class change {drop}"))
    return verdicts


def cmd_decide(args) -> int:
    tests = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "test_*.json"))):
        with open(path) as f:
            tests.append(json.load(f))
    for rule, text in decide(tests):
        print(f"{rule:40s} {text}")
    return 0 if tests else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "test", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[], help="name:kind:trained_fold:path")
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    args = p.parse_args(argv)
    if args.stage == "decide":
        return cmd_decide(args)
    if not args.data_path or not args.checkpoint:
        p.error("select and test need --data_path and at least one --checkpoint")
    device = torch.device("cuda")
    return cmd_select(args, device) if args.stage == "select" else cmd_test(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
