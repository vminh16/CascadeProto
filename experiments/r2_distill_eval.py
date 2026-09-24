"""R2 of [DECISION D-29]: oracle-direction distillation against its reference, scored on identical episodes.

    python experiments/r2_distill_eval.py test --data_path datasets/S3DIS/blocks_bs1_s1 --cvfold 1 \
        --checkpoint r0:ours:1:log_r2/.../last.pt --checkpoint d29:ours:1:log_r2/.../last.pt \
        --checkpoint vipseg:vipseg:1:vipseg_S1_N2_K1.pt
    python experiments/r2_distill_eval.py decide --cvfold 1                       # rules R2.0-R2.5
    python experiments/r2_distill_eval.py decide_e1 --cvfold 1 --out_dir results/phase16_e1   # D-30
    python experiments/r2_distill_eval.py decide_n1 --cvfold 1 --out_dir results/phase16_n1   # D-33

Each arm is trained once (maintainer, 2026-09-23), so the test repeats over draws instead: fixed100 (the
table's protocol, one cached draw) and three independent `random600` draws (seeds 0, 1, 2). Every model
is scored on the **same episodes** of a draw, which makes each comparison paired: the paired bootstrap over
episodes gives the uncertainty of the draw, not the training-seed noise (D-29 estimates that at sd ≈ 0.5
for the difference of two single runs, from R1's `r1_vippem` seeds).

Mechanism diagnostics, query labels used for diagnostics only: the logit-pair cosine to the oracle rule
that D-29 trains (02 §14), the prototype-space cosines `cos(M_eff, O)` and `cos(M_c - M_c', O_c - O_c')`
per step (descriptive: see `models/oracle_distill.py` for what they cannot see), and the mIoU with the
present classes of `M_eff` replaced by `O`, norms kept (P0's `ORACLE_REPLACE`) or equal (the rule D-29
trains toward): the headroom a model leaves.
Every checkpoint passes eval.py's protocol guard [DECISION D-22].
"""

import argparse
import glob
import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments.p0_em_probe import (VIPSegScoringRule, check_identity, class_ious, episode_counts,  # noqa: E402
                                     miou_from_counts, paired_bootstrap, parse_checkpoint)
from models.oracle_distill import (cosine_to_oracle, oracle_directions, oracle_logits,  # noqa: E402
                                   pair_logit_cosine, pairwise_cosine)

OUT_DIR = "results/phase16_r2"
DRAWS = ("fixed100", "random600:0", "random600:1", "random600:2")
REFERENCE, ARM, RELEASED = "r0", "d29", "vipseg"
E1, E1_BEST, REFERENCE_BEST = "e1", "e1_best", "r0_best"  # [DECISION D-30]
# (a, b) -> paired "b_vs_a", computed when both names are scored
PAIRS = ((REFERENCE, ARM), (RELEASED, REFERENCE), (RELEASED, ARM),
         (REFERENCE, E1), (RELEASED, E1), (RELEASED, E1_BEST), (REFERENCE_BEST, E1_BEST), (E1, E1_BEST),
         (REFERENCE, REFERENCE_BEST))
E1_ADOPT, E1_NOT_SCHEDULE = 73.0, 71.0  # VIP-Seg's own last-update level; r0 + our last.pt spread [D-30]
CTL, NECK = "ctl", "neck"  # [DECISION D-33]
PAIRS = PAIRS + ((CTL, NECK), (E1, CTL), (E1, NECK), (CTL + "_best", NECK + "_best"))
GO_GAIN, STOP_GAIN = 1.0, 0.5  # 2x and 1x the estimated sd of a single-run difference [DECISION D-29]
REFERENCE_GAP = -2.0  # R2.0: our loop's head below VIP-Seg's own by more than this [DECISION D-29]
COLLAPSE = -3.0  # R2.5, per-class IoU points, as P0-P2


# ------------------------------------------------------------------ models behind one call

class OursRule:
    """F^q, M_eff and the stage prototypes of a CascadeProto checkpoint, read through `model.cascade`."""

    def __init__(self, model):
        self.model = model

    def __call__(self, episode) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        f_q, p0, steps, _ = self.model.cascade(episode)  # [B_q,P,D], [N+1,D], T x [B_q,N+1,D]
        m_eff = self.model.effective_prototype(f_q, p0, steps)  # [B_q, N+1, D]
        if self.model.config.logit_scale == "sqrt_D":
            m_eff = m_eff / math.sqrt(f_q.shape[-1])  # [B_q, N+1, D], folds the scale into the prototype
        logits = self.model(episode).logits  # [B_q, P, N+1]
        return f_q, m_eff, [p0.unsqueeze(0).expand_as(m_eff)] + list(steps), logits

    def close(self):
        pass


class ReleasedRule:
    """VIP-Seg's released model through P0's hooks; its steps are the prototypes after each module."""

    def __init__(self, baseline):
        self.rule = VIPSegScoringRule(baseline)

    def __call__(self, episode):
        f_q, m_eff, logits = self.rule(episode)  # [B_q,P,D], [B_q,N+1,D], [B_q,P,N+1]
        r = self.rule
        steps = [r.step_in[0]] + [out + inp if t % 2 == 1 else out  # outer residual on PDM steps
                                  for t, (inp, out) in enumerate(zip(r.step_in, r.step_out))]
        return f_q, m_eff, steps, logits

    def close(self):
        self.rule.close()


def load(ck: SimpleNamespace, cvfold: int, device):
    import eval as eval_cli

    args = SimpleNamespace(checkpoint=ck.path, model="vipseg" if ck.kind == "vipseg" else "cascadeproto",
                           cvfold=cvfold, checkpoint_cvfold=ck.fold, allow_seen_classes=False)
    model = eval_cli.load_model(args, device)
    protocol = eval_cli.protocol_check(args, model.checkpoint_args)  # raises on seen classes [D-22]
    model.eval()
    rule = ReleasedRule(model) if ck.kind == "vipseg" else OursRule(model)
    config = model.config.to_dict() if hasattr(model, "config") else None
    return rule, protocol, config


# ------------------------------------------------------------------ per episode

def oracle_replaced(f_q: torch.Tensor, m_eff: torch.Tensor, labels: torch.Tensor, unit: bool = False) -> torch.Tensor:
    """Predictions [B_q, P] with every present class of M_eff replaced by O; absent classes untouched.

    unit=False keeps each class's norm (P0's ORACLE_REPLACE rule). unit=True gives the present classes one
    common norm, their mean, so the rule depends on the directions O alone: the rule whose pairwise
    hyperplanes are those of `O_c - O_c'` [DECISION D-29].
    """
    o, present = oracle_directions(f_q, labels, m_eff.shape[1])  # [B_q, N+1, D], [B_q, N+1]
    norm = m_eff.norm(dim=-1, keepdim=True)  # [B_q, N+1, 1]
    if unit:
        pres = present.unsqueeze(-1).to(norm.dtype)  # [B_q, N+1, 1]
        norm = ((norm * pres).sum(dim=1, keepdim=True) / pres.sum(dim=1, keepdim=True)).expand_as(norm)  # [B_q,N+1,1]
    m = torch.where(present.unsqueeze(-1), norm * o, m_eff)  # [B_q, N+1, D]
    return torch.einsum("bpd,bcd->bpc", f_q, m).argmax(dim=-1)  # [B_q, P]


def cosine_terms(f_q, m_eff, steps, labels, logits) -> Dict[str, np.ndarray]:
    """Sums and counts over present classes / pairs.

    `logit_pair`: cos_i(L_c - L_c', T_c - T_c'), the quantity D-29 trains, free of the common shift, of the
    scale and of components orthogonal to the features; the mechanism rule reads it [DECISION D-29].
    Prototype space, descriptive only: raw cos(., O) (`bg`, `fg`, `step<t>`), which the common shift
    moves, and cos(m_c - m_c', O_c - O_c') (`pair`, `pair_step<t>`), which the orthogonal components move.
    """
    lc, lm = pair_logit_cosine(logits, oracle_logits(f_q, labels, logits.shape[-1]), labels)  # [B_q,N+1,N+1]
    cos, present = cosine_to_oracle(m_eff, f_q, labels)  # [B_q, N+1], [B_q, N+1]
    o, _ = oracle_directions(f_q, labels, m_eff.shape[1])  # [B_q, N+1, D]
    bg, fg = present[:, 0], present[:, 1:]  # [B_q], [B_q, N]
    out = {"logit_pair": np.array([float(lc[lm].sum()), float(lm.sum())]),
           "bg": np.array([float(cos[:, 0][bg].sum()), float(bg.sum())]),
           "fg": np.array([float(cos[:, 1:][fg].sum()), float(fg.sum())])}
    pc, pm = pairwise_cosine(m_eff, o, present)  # [B_q, N+1, N+1]
    out["pair"] = np.array([float(pc[pm].sum()), float(pm.sum())])
    for t, p in enumerate(steps):
        c, _ = cosine_to_oracle(p, f_q, labels)  # [B_q, N+1]
        out[f"step{t}"] = np.array([float(c[present].sum()), float(present.sum())])
        pc, pm = pairwise_cosine(p, o, present)  # [B_q, N+1, N+1]
        out[f"pair_step{t}"] = np.array([float(pc[pm].sum()), float(pm.sum())])
    return out


def episodes_of(draw: str, data_path: str, cvfold: int):
    """(dataset, test classes) of a draw; random600:s re-seeds the loader's global RNGs with s (eval.py)."""
    from dataloaders.loader import MyDataset
    from pipeline.episodes import (N_QUERIES, NUM_POINT, PC_ATTRIBS, WAY_NUM, WAY_RATIO, build_eval_dataset)
    from train import seed_everything

    if draw == "fixed100":
        ds = build_eval_dataset(data_path, "s3dis", cvfold, 2, 1, mode="test", seed=0)
    elif draw.startswith("random600:"):
        seed_everything(int(draw.split(":")[1]))
        ds = MyDataset(data_path, "s3dis", cvfold=cvfold, num_episode=600, n_way=2, k_shot=1, n_queries=N_QUERIES,
                       mode="test", num_point=NUM_POINT, pc_attribs=PC_ATTRIBS, pc_augm=False,
                       way_ratio=WAY_RATIO, way_num=WAY_NUM)
    else:
        raise ValueError(f"unknown draw {draw!r}; expected one of {DRAWS}")
    return ds, [int(c) for c in np.asarray(ds.classes)]


@torch.no_grad()
def score_draw(rules: Dict[str, object], draw: str, data_path: str, cvfold: int, device, max_episodes=None):
    from torch.utils.data import DataLoader, Subset

    from pipeline.episodes import EpisodeCollate, read_class_names

    names = read_class_names(data_path, "s3dis")
    ds, test_classes = episodes_of(draw, data_path, cvfold)
    view = ds if max_episodes is None else Subset(ds, range(min(max_episodes, len(ds))))
    loader = DataLoader(view, batch_size=1, shuffle=False, collate_fn=EpisodeCollate(names))  # main process
    counts: Dict[str, List[np.ndarray]] = {}
    cos: Dict[str, Dict[str, np.ndarray]] = {n: {} for n in rules}
    preds_of = {n: [] for n in rules}
    gts, l2c = [], []
    for (episode,) in loader:
        episode = episode.to(device)
        labels = episode.query_y  # [B_q, P], diagnostics and the oracle arm only
        gt = labels.cpu().numpy()
        for name, rule in rules.items():
            f_q, m_eff, steps, logits = rule(episode)
            check_identity(f_q, m_eff, logits)  # the rule read the tensors the model scores with
            pred = logits.argmax(dim=-1).cpu().numpy()  # [B_q, P]
            preds_of[name].append(pred)
            for key, p in ((name, pred), (name + "_oracle", oracle_replaced(f_q, m_eff, labels).cpu().numpy()),
                           (name + "_oracle_unit", oracle_replaced(f_q, m_eff, labels, unit=True).cpu().numpy())):
                counts.setdefault(key, []).append(episode_counts(p, gt, episode.sampled_classes, test_classes))
            for key, v in cosine_terms(f_q, m_eff, steps, labels, logits).items():
                cos[name][key] = cos[name].get(key, 0.0) + v
        gts.append(gt), l2c.append(episode.sampled_classes)
    stacked = {k: np.stack(v) for k, v in counts.items()}  # name -> [E, 3, C+1]
    if draw == "fixed100":  # the count-based mIoU must be VIP-Seg's own metric [D-08]
        from pipeline.evaluation import accumulated_miou

        class _Quiet:
            def cprint(self, text):
                pass

        for name in rules:
            primary = accumulated_miou(_Quiet(), preds_of[name], gts, l2c, test_classes)
            if abs(primary - miou_from_counts(stacked[name].sum(0))) > 1e-9:
                raise RuntimeError(f"{name}: count-based mIoU differs from VIP-Seg's evaluate_metric")
    result = {"draw": draw, "cvfold": cvfold, "episodes": len(view), "test_classes": test_classes,
              "miou": {k: float(miou_from_counts(v.sum(0))) for k, v in stacked.items()},
              "class_iou": {k: class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
              "cos": {n: {k: float(v[0] / max(v[1], 1.0)) for k, v in c.items()} for n, c in cos.items()}}
    pairs = [(a, b) for a, b in PAIRS if a in rules and b in rules]
    result["paired"] = {f"{b}_vs_{a}": paired_bootstrap(stacked[a], stacked[b]) for a, b in pairs}
    result["paired"].update({f"{n}{o}_vs_{n}": paired_bootstrap(stacked[n], stacked[n + o])
                             for n in rules for o in ("_oracle", "_oracle_unit")})
    return result, stacked


def draw_stem(cvfold: int, draw: str) -> str:
    return f"test_S{cvfold}_{draw.replace(':', '_seed')}"


def cmd_test(args, device) -> int:
    rules, meta = {}, {}
    for spec in args.checkpoint:
        ck = parse_checkpoint(spec)
        if ck.fold != args.cvfold:
            raise ValueError(f"{ck.name} was trained on S{ck.fold}; this run scores S{args.cvfold} [DECISION D-22]")
        rules[ck.name], protocol, config = load(ck, args.cvfold, device)
        meta[ck.name] = {"checkpoint": vars(ck), "protocol": protocol, "config": config}
    os.makedirs(args.out_dir, exist_ok=True)
    try:  # the rules (VIP-Seg's hooks) serve every draw and are closed once, at the end
        for draw in args.draws:
            result, stacked = score_draw(rules, draw, args.data_path, args.cvfold, device, args.max_episodes)
            result["models"] = meta
            stem = draw_stem(args.cvfold, draw)
            with open(os.path.join(args.out_dir, stem + ".json"), "w") as f:
                json.dump(result, f, indent=1)
            np.savez_compressed(os.path.join(args.out_dir, stem + "_counts.npz"), **stacked)
            p = result["paired"].get(f"{ARM}_vs_{REFERENCE}")
            print(f"[test] S{args.cvfold} {draw}: " + " | ".join(
                f"{n} {result['miou'][n]:.4f} (logit-pair cos {result['cos'][n]['logit_pair']:.3f}, oracle "
                f"{result['miou'][n + '_oracle']:.4f} / unit {result['miou'][n + '_oracle_unit']:.4f})" for n in rules)
                + (f" | {ARM} - {REFERENCE} {p['gain']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}]" if p else ""),
                flush=True)
    finally:
        for rule in rules.values():
            rule.close()
    return 0


# ------------------------------------------------------------------ rules

def ci_text(p: Dict) -> str:
    side = "above 0" if p["ci_low"] > 0 else "below 0" if p["ci_high"] < 0 else "contains 0"
    return f"{p['gain']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}], {side}"


def decide(results: List[Dict], cvfold: int) -> List[Tuple[str, str]]:
    """Rules R2.0-R2.5 of [DECISION D-29] on the draws of one fold."""
    by_draw = {r["draw"]: r for r in results if r["cvfold"] == cvfold}
    missing = [d for d in DRAWS if d not in by_draw]
    if missing:
        return [("incomplete", f"S{cvfold}: missing draws {missing}")]
    fixed = by_draw["fixed100"]
    if REFERENCE not in fixed["miou"] or ARM not in fixed["miou"]:
        return [("incomplete", f"S{cvfold}: {REFERENCE!r} and {ARM!r} must both be scored")]
    verdicts = []
    if RELEASED in fixed["miou"]:
        gap = 100.0 * (fixed["miou"][REFERENCE] - fixed["miou"][RELEASED])
        verdicts.append(("R2.0 reference", f"{REFERENCE} - VIP-Seg released {gap:+.2f} on fixed100: " +
                         ("our loop trains the head worse; a gain over R0 is not a gain over VIP-Seg"
                          if gap < REFERENCE_GAP else "comparable")))
    p = fixed["paired"][f"{ARM}_vs_{REFERENCE}"]
    rand = [100.0 * (by_draw[d]["miou"][ARM] - by_draw[d]["miou"][REFERENCE]) for d in DRAWS[1:]]
    lp = {n: fixed["cos"][n]["logit_pair"] for n in (REFERENCE, ARM)}
    mech = lp[ARM] > lp[REFERENCE]
    mech_text = f"logit-pair cosine to the oracle rule {lp[REFERENCE]:.4f} -> {lp[ARM]:.4f}"
    gain_text = f"fixed100 {ci_text(p)}; random600 {[round(g, 2) for g in rand]}; {mech_text}"
    if p["gain"] < STOP_GAIN or float(np.mean(rand)) < STOP_GAIN:
        verdicts.append(("R2.2 stop", gain_text))
    elif p["gain"] >= GO_GAIN and p["ci_low"] > 0 and min(rand) > 0 and mech:
        nxt = ("train both arms on S0 once each, same test" if cvfold == 1 else
               f"S0 confirmed; beats VIP-Seg's 72.20: {'yes' if 100.0 * fixed['miou'][ARM] > 72.20 else 'no'}")
        verdicts.append(("R2.1 go", f"{gain_text}: {nxt}"))
    elif p["gain"] >= GO_GAIN and p["ci_low"] > 0 and min(rand) > 0:
        verdicts.append(("R2.4 mechanism", f"{gain_text}: a gain without a higher logit-pair cosine is not a "
                                           f"distillation result; treated as R2.3"))
    else:
        verdicts.append(("R2.3 in between", f"{gain_text}: one run per arm cannot separate this from training "
                                            f"noise; a second training seed per arm is needed"))
    drop = [round(100.0 * (b - a), 2) for a, b in zip(fixed["class_iou"][REFERENCE][1:], fixed["class_iou"][ARM][1:])]
    if p["gain"] > 0 and min(drop) < COLLAPSE:
        verdicts.append(("R2.5 collapse watch", f"per-class change {drop} (classes {fixed['test_classes']})"))
    for name in (REFERENCE, ARM, RELEASED):
        if name in fixed["miou"]:
            o, u = (fixed["paired"][f"{name}{k}_vs_{name}"]["gain"] for k in ("_oracle", "_oracle_unit"))
            verdicts.append((f"headroom {name}", f"oracle replacement {o:+.2f}, oracle directions {u:+.2f} "
                                                 f"on fixed100; logit-pair cosine {fixed['cos'][name]['logit_pair']:.4f}"))
    return verdicts


def decide_e1(results: List[Dict], cvfold: int) -> List[Tuple[str, str]]:
    """Rules E1.1-E1.3 of [DECISION D-30] on the draws of one fold, plus the reported comparisons."""
    by_draw = {r["draw"]: r for r in results if r["cvfold"] == cvfold}
    missing = [d for d in DRAWS if d not in by_draw]
    if missing:
        return [("incomplete", f"S{cvfold}: missing draws {missing}")]
    fixed = by_draw["fixed100"]
    if E1 not in fixed["miou"] or REFERENCE not in fixed["miou"]:
        return [("incomplete", f"S{cvfold}: {E1!r} and {REFERENCE!r} must both be scored")]
    last = 100.0 * fixed["miou"][E1]
    p = fixed["paired"][f"{E1}_vs_{REFERENCE}"]
    rand = [100.0 * (by_draw[d]["miou"][E1] - by_draw[d]["miou"][REFERENCE]) for d in DRAWS[1:]]
    text = f"E1 last {last:.2f}; E1 - r0 fixed100 {ci_text(p)}; random600 {[round(g, 2) for g in rand]}"
    if last >= E1_ADOPT:
        verdicts = [("E1.1 adopt", f"{text}: VIP-Seg's update count becomes route B's schedule")]
    elif last <= E1_NOT_SCHEDULE:
        verdicts = [("E1.2 not the schedule", f"{text}: the gap lies elsewhere")]
    elif p["ci_low"] > 0 and min(rand) > 0:
        verdicts = [("E1.3 partial, adopt", f"{text}: paired gain holds on every draw")]
    else:
        verdicts = [("E1.3 partial, keep", f"{text}: the gain does not hold on every draw")]
    for a, b in ((RELEASED, E1), (RELEASED, E1_BEST), (E1, E1_BEST), (REFERENCE, REFERENCE_BEST)):
        key = f"{b}_vs_{a}"
        if key in fixed["paired"]:
            verdicts.append((f"reported {key}", ci_text(fixed["paired"][key])))
    return verdicts


def decide_n1(results: List[Dict], cvfold: int) -> List[Tuple[str, str]]:
    """Rules N1.1-N1.4 of [DECISION D-33] on the draws of one fold."""
    by_draw = {r["draw"]: r for r in results if r["cvfold"] == cvfold}
    missing = [d for d in DRAWS if d not in by_draw]
    if missing:
        return [("incomplete", f"S{cvfold}: missing draws {missing}")]
    fixed = by_draw["fixed100"]
    if CTL not in fixed["miou"] or NECK not in fixed["miou"]:
        return [("incomplete", f"S{cvfold}: {CTL!r} and {NECK!r} must both be scored")]
    p = fixed["paired"][f"{NECK}_vs_{CTL}"]
    rand = [100.0 * (by_draw[d]["miou"][NECK] - by_draw[d]["miou"][CTL]) for d in DRAWS[1:]]
    text = f"neck - ctl fixed100 {ci_text(p)}; random600 {[round(g, 2) for g in rand]}"
    if p["gain"] < STOP_GAIN or float(np.mean(rand)) < STOP_GAIN:
        v = [("N1.2 stop", text)]
    elif p["gain"] >= GO_GAIN and p["ci_low"] > 0 and min(rand) > 0:
        v = [("N1.1 go", f"{text}: S0 next")]
    else:
        v = [("N1.3 in between", text)]
    heads = {n: fixed["paired"][f"{n}_oracle_unit_vs_{n}"]["gain"] for n in (CTL, NECK)
             if f"{n}_oracle_unit_vs_{n}" in fixed["paired"]}
    flag = " (a go with an unchanged gap is flagged)" if v[0][0] == "N1.1 go" and heads.get(NECK, 0) >= heads.get(CTL, 0) else ""
    v.append(("N1.4 oracle gap", ", ".join(f"{n} {g:+.2f}" for n, g in heads.items()) + flag))
    for a, b in ((E1, CTL), (E1, NECK), (CTL + "_best", NECK + "_best")):
        key = f"{b}_vs_{a}"
        if key in fixed["paired"]:
            v.append((f"reported {key}", ci_text(fixed["paired"][key])))
    return v


def load_results(out_dir: str, cvfold: int) -> List[Dict]:
    results = []
    for path in sorted(glob.glob(os.path.join(out_dir, f"test_S{cvfold}_*.json"))):
        with open(path) as f:
            results.append(json.load(f))
    return results


def cmd_decide(args) -> int:
    results = load_results(args.out_dir, args.cvfold)
    rules = {"decide_e1": decide_e1, "decide_n1": decide_n1}.get(args.stage, decide)
    for rule, text in rules(results, args.cvfold):
        print(f"{rule:24s} {text}")
    return 0 if results else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["test", "decide", "decide_e1", "decide_n1"])
    p.add_argument("--data_path")
    p.add_argument("--cvfold", type=int, required=True, choices=[0, 1])
    p.add_argument("--checkpoint", action="append", default=[], help="name:kind:trained_fold:path")
    p.add_argument("--draws", nargs="+", default=list(DRAWS), choices=list(DRAWS))
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--out_dir", default=OUT_DIR, help="results/phase16_e1 for D-30, so R2's files stay intact")
    args = p.parse_args(argv)
    if args.stage in ("decide", "decide_e1", "decide_n1"):
        return cmd_decide(args)
    if not args.data_path or not args.checkpoint:
        p.error("test needs --data_path and the checkpoints")
    return cmd_test(args, torch.device("cuda"))


if __name__ == "__main__":
    raise SystemExit(main())
