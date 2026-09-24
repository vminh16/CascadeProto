"""D-37: the 2 x 2 of head (scrambled, clean) x training query order (fixed, random): leak-free draw and rules.

The standard draws (fixed100, random600 seeds 0-2, with oracle columns) are scored by
`experiments/r2_distill_eval.py test --out_dir results/phase16_d37`, the swapped order by
`experiments/c3_query_order.py --out results/phase16_d37/c3_swap.json`; this script adds P5's leak-free draw
(arm D, seed 4) and reads everything against the rules of [DECISION D-37].

Arms (names on the command line): `e1` / `e1_best` = VF (existing E1 checkpoints), `vr` / `vr_best` = scrambled
head, random order, `cr` / `cr_best` = clean head, random order. CF = CR by D-37's lemma (tests D37-T5, D37-T6).

    python experiments/d37_eval.py leakfree --data_path datasets/S3DIS/blocks_bs1_s1 --checkpoint vr:ours:1:<path> ...
    python experiments/d37_eval.py decide
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402

OUT_DIR = "results/phase16_d37"
E1_DIR = "results/phase16_e1"
SWAP_JSON = "c3_swap.json"
DRAWS = r2.DRAWS  # fixed100, random600:0, random600:1, random600:2
VF, VR, CR = "e1", "vr", "cr"
E1_TOL = 0.05  # mIoU points: the re-scored E1 must reproduce E1's own evaluation
ORIGIN_GAP, ORIGIN_RELABEL, STILL_POSITIONAL = 1.0, 0.05, 0.5  # D37.1
EQUI_GAP, EQUI_RELABEL = 0.1, 0.01  # D37.2
ARCH_GAIN = 1.0  # D37.3, twice the single-run sd (D-29)


def stem(draw: str) -> str:
    return r2.draw_stem(1, draw)


def load_draws(out_dir: str) -> Dict[str, Tuple[Dict, Dict[str, np.ndarray]]]:
    found = {}
    for draw in DRAWS:
        path = os.path.join(out_dir, stem(draw) + ".json")
        if os.path.isfile(path):
            with open(path) as f:
                result = json.load(f)
            counts = dict(np.load(os.path.join(out_dir, stem(draw) + "_counts.npz")))
            found[draw] = (result, counts)
    return found


def swap_numbers(swap: Dict, name: str) -> Tuple[float, float, float]:
    """(stored mIoU, swapped mIoU, relabel shift) of one checkpoint in C3's output, in points and share.

    The shift is the own-class share predicted as the other episode class (the new position's label) in the swapped
    order minus the same share in the stored order: 0 for an order-free model [DECISION D-36].
    """
    r = swap["checkpoints"][name]
    s = r["own_class_share"]
    return 100.0 * r["miou"]["model"], 100.0 * r["miou"]["model_swapped"], s["swap_position"] - s["normal_other"]


def paired(counts: Dict[str, np.ndarray], a: str, b: str) -> Dict[str, float]:
    """b - a in mIoU points on the same episodes, bootstrap over episodes."""
    return p0.paired_bootstrap(counts[a], counts[b])


def e1_reproduced(draws: Dict, e1_draws: Dict) -> List[Tuple[str, str, bool]]:
    """The re-scored E1 checkpoints against E1's own evaluation, draw by draw."""
    out = []
    for draw in DRAWS:
        if draw not in draws or draw not in e1_draws:
            out.append((draw, "missing", False))
            continue
        for name in (VF, VF + "_best"):
            now, then = draws[draw][0]["miou"].get(name), e1_draws[draw][0]["miou"].get(name)
            if now is None or then is None:
                out.append((f"{draw} {name}", "missing", False))
                continue
            d = 100.0 * abs(now - then)
            out.append((f"{draw} {name}", f"{100 * now:.2f} vs {100 * then:.2f} (|diff| {d:.3f})", d <= E1_TOL))
    return out


def decide(draws: Dict, swap: Optional[Dict], leakfree: Dict[str, Dict], e1_draws: Dict,
           suffix: str = "") -> List[Tuple[str, str]]:
    """Rules D37.1-D37.4 on the `last` checkpoints (suffix "") or the `best` ones (suffix "_best")."""
    vf, vr, cr = VF + suffix, VR + suffix, CR + suffix
    missing = [d for d in DRAWS if d not in draws]
    if missing or swap is None:
        return [("incomplete", f"missing draws {missing}" + ("" if swap is not None else f", missing {SWAP_JSON}"))]
    check = e1_reproduced(draws, e1_draws)
    bad = [f"{k}: {t}" for k, t, ok in check if not ok]
    if bad:
        return [("E1 check failed", "the evaluation does not reproduce E1's; no rule is read: " + "; ".join(bad))]
    fixed = draws["fixed100"]
    if any(n not in fixed[0]["miou"] for n in (vf, vr, cr)) or any(n not in swap["checkpoints"] for n in (vf, vr, cr)):
        return [("incomplete", f"{vf}, {vr}, {cr} must all be scored on every draw and in {SWAP_JSON}")]
    verdicts = []
    # D37.2 first: if the clean head is not order-free nothing else can be read
    s, w, rel = swap_numbers(swap, cr)
    if abs(s - w) > EQUI_GAP or abs(rel) > EQUI_RELABEL:
        return [("D37.2 FAILED", f"{cr} stored {s:.2f}, swapped {w:.2f}, relabel shift {rel:+.4f}: the clean head is not "
                                 f"equivariant; no other rule is read")]
    verdicts.append(("D37.2 equivariance", f"{cr} stored {s:.2f}, swapped {w:.2f}, relabel shift {rel:+.4f}"))
    # D37.1 origin
    s, w, rel = swap_numbers(swap, vr)
    text = f"{vr} stored {s:.2f}, swapped {w:.2f} (gap {abs(s - w):.2f}), relabel shift {rel:+.3f}"
    if abs(s - w) <= ORIGIN_GAP and rel <= ORIGIN_RELABEL:
        verdicts.append(("D37.1 origin", f"{text}: the loader's fixed order is the origin of the shortcut"))
    elif rel > STILL_POSITIONAL:
        verdicts.append(("D37.1 STOP", f"{text}: the scrambled head still names classes by position without a "
                                       f"positional signal in training; stop and re-examine"))
    else:
        verdicts.append(("D37.1 in between", f"{text}: partly positional; reported, D37.3 read with this caveat"))
    # D37.3 architecture, clean vs scrambled with no shortcut available
    p = paired(fixed[1], vr, cr)
    rand = [100.0 * (draws[d][0]["miou"][cr] - draws[d][0]["miou"][vr]) for d in DRAWS[1:]]
    text = f"{cr} - {vr} fixed100 {r2.ci_text(p)}; random600 {[round(g, 2) for g in rand]}"
    if p["gain"] >= ARCH_GAIN and p["ci_low"] > 0 and min(rand) > 0:
        verdicts.append(("D37.3 clean head", f"{text}: the clean head is the base"))
    elif p["gain"] <= -ARCH_GAIN and p["ci_high"] < 0 and max(rand) < 0:
        verdicts.append(("D37.3 scrambled head", f"{text}: the scrambled head, trained with random order, is the base"))
    else:
        verdicts.append(("D37.3 tie -> clean head", f"{text}: no difference by the rule; the clean head is the base "
                                                    f"(cannot re-learn the shortcut)"))
    base = cr if verdicts[-1][0] != "D37.3 scrambled head" else vr
    # D37.4 reported
    p = paired(fixed[1], vr, vf)
    rand = [100.0 * (draws[d][0]["miou"][vf] - draws[d][0]["miou"][vr]) for d in DRAWS[1:]]
    verdicts.append(("D37.4 shortcut worth", f"{vf} - {vr} fixed100 {r2.ci_text(p)}; random600 "
                                             f"{[round(g, 2) for g in rand]}"))
    for n in (vf, vr, cr):
        s, w, rel = swap_numbers(swap, n)
        lf = leakfree.get(n)
        lf_text = (f"leak-free {100 * lf['miou']['model']:.2f} (support rule {100 * lf['miou']['support_rule']:.2f}, "
                   f"oracle {100 * lf['miou']['oracle_unit']:.2f}, {lf['episodes']} episodes)" if lf else "leak-free missing")
        verdicts.append((f"D37.4 {n}", f"fixed100 stored {s:.2f} / swapped {w:.2f}, relabel shift {rel:+.3f}; {lf_text}"))
    for key in ("_oracle", "_oracle_unit"):
        verdicts.append((f"D37.4 oracle gap{key}", f"{base}: {r2.ci_text(fixed[0]['paired'][f'{base}{key}_vs_{base}'])} "
                                                   f"on fixed100"))
    return verdicts


@torch.no_grad()
def cmd_leakfree(args, device) -> int:
    from experiments import p5_condition_probe as p5

    os.makedirs(args.out_dir, exist_ok=True)
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        if ck.fold != 1:
            raise ValueError(f"{ck.name}: D-37 runs on S1 only [DECISION D-22]")
        rule, protocol, config = r2.load(ck, 1, device)
        try:
            result, stacked = p5.score_leakfree(rule, args.data_path, device, args.per_pair, args.max_episodes)
        finally:
            rule.close()
        result.update(name=ck.name, protocol=protocol, config=config, checkpoint=vars(ck))
        tag = "" if args.max_episodes is None else "_smoke"
        with open(os.path.join(args.out_dir, f"leakfree_{ck.name}{tag}.json"), "w") as f:
            json.dump(result, f, indent=1)
        np.savez_compressed(os.path.join(args.out_dir, f"leakfree_{ck.name}{tag}_counts.npz"), **stacked)
        print(f"[leakfree] {ck.name}: " + " | ".join(f"{k} {100 * v:.2f}" for k, v in result["miou"].items())
              + f" | {result['episodes']} episodes, {result['skipped']} skipped", flush=True)
    return 0


def cmd_decide(args) -> int:
    draws, e1_draws = load_draws(args.out_dir), load_draws(args.e1_dir)
    swap_path = os.path.join(args.out_dir, SWAP_JSON)
    swap = json.load(open(swap_path)) if os.path.isfile(swap_path) else None
    leakfree = {}
    for path in glob.glob(os.path.join(args.out_dir, "leakfree_*.json")):
        if not path.endswith("_smoke.json"):
            with open(path) as f:
                r = json.load(f)
            leakfree[r["name"]] = r
    for key, text, ok in e1_reproduced(draws, e1_draws):
        print(f"{'E1 check':24s} {key}: {text}{'' if ok else '  <-- FAILED'}")
    for suffix, title in (("", "last.pt (headline, D-22)"), ("_best", "best.pt (best of validation)")):
        print(f"--- {title}")
        for rule, text in decide(draws, swap, leakfree, e1_draws, suffix):
            print(f"{rule:24s} {text}")
    return 0 if draws else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["leakfree", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[], help="name:ours:1:path")
    p.add_argument("--per_pair", type=int, default=100)
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--e1_dir", default=E1_DIR)
    args = p.parse_args(argv)
    if args.stage == "decide":
        return cmd_decide(args)
    if not args.data_path or not args.checkpoint:
        p.error("leakfree needs --data_path and the checkpoints")
    return cmd_leakfree(args, torch.device("cuda"))


if __name__ == "__main__":
    raise SystemExit(main())
