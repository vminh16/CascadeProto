"""C4 of [DECISION D-36]: is VIP-Seg's cross-term reshape the only carrier of the query position? (GPU, inference)

Each checkpoint is re-run on the fixed100 episodes with the cross-term of every PEM/PDM in three forms, reading
the same trained weights (`models/vip_stage.py`): `native` (the inherited module, as trained), `scrambled` (the
written-out forward with VIP-Seg's `reshape(72, -1)`, which must reproduce `native`) and `clean` (one attention
per query and slot). `native` and `clean` are scored in the stored order and with the two query blocks swapped
(C3's scoring, labels moving with their blocks, predictions mapped back). VIP-Seg's released model is re-run with
the entries of its `vip_module` wrapped in the loaded instance; no inherited file or class is edited.

Checks (a failure raises): C4.0a `scrambled` reproduces `native` on every episode; C4.0b the native passes
reproduce C3's mIoU (`results/phase16_p5/c3_query_order.json`) within 0.01. Rule C4.1 (`decide`): with `clean`,
|stored - swapped| <= 0.1 mIoU and |relabel shift| <= 0.01 on every checkpoint (the own-class share predicted as the
other episode class, swapped minus stored); exit code 3 otherwise, so that D-37's queue stops.

    python experiments/c4_crossterm_ablation.py --data_path datasets/S3DIS/blocks_bs1_s1 \
        --checkpoint e1:ours:1:log_r2/s3dis_S1_N2_K1_point_T4_vip_b1/last.pt --checkpoint vipseg:vipseg:1:vipseg_S1_N2_K1.pt
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from experiments import p0_em_probe as p0  # noqa: E402
from experiments import r2_distill_eval as r2  # noqa: E402
from experiments.c3_query_order import swapped  # noqa: E402

OUT = "results/phase16_d37/c4_crossterm.json"
C3_JSON = "results/phase16_p5/c3_query_order.json"
PASSES = (("native", False), ("native", True), ("scrambled", False), ("clean", False), ("clean", True))
FORM_RTOL = 1e-4  # C4.0a, relative to max(1, max |native logit|) [DECISION D-36]
C3_TOL = 0.01  # C4.0b, mIoU points
SWAP_GAP, RELABEL = 0.1, 0.01  # C4.1


def pass_name(form: str, swap: bool) -> str:
    return f"{form}_{'swapped' if swap else 'stored'}"


def own_update(own: Dict[str, int], pred: np.ndarray, qy: np.ndarray, swap: bool) -> None:
    """Add one episode's own-class counts: pred and qy [2, P] in the stored order (C3's bookkeeping).

    Query block b was sampled for local class b + 1; after a swap it sat at position 1 - b, whose label 2 - b is the
    other episode class's. `other` counts that label in both orders, so that the relabel shift (swapped minus stored)
    separates the position from honest confusion [DECISION D-36].
    """
    if qy.shape[0] != 2:
        raise ValueError(f"C4 reads 2-way episodes with one query block per way, got {qy.shape[0]} blocks")
    for b in range(2):
        m = qy[b] == b + 1  # [P], the block's own (sampled-for) class
        own["points"] += int(m.sum())
        own["own"] += int((pred[b][m] == b + 1).sum())
        own["other"] += int((pred[b][m] == 2 - b).sum())  # = the new position's label when swapped
        own["background"] += int((pred[b][m] == 0).sum())


def shares(own: Dict[str, int]) -> Dict[str, float]:
    n = max(own["points"], 1)
    return {k: v / n for k, v in own.items() if k != "points"}


def relabel_shift(result: Dict, form: str) -> float:
    """Own-class share predicted as the other episode class, swapped minus stored order [DECISION D-36]."""
    s = result["own_class_share"]
    return s[f"{form}_swapped"]["other"] - s[f"{form}_stored"]["other"]


def load_with_forms(ck: SimpleNamespace, device):
    """(scoring rule, protocol, set_form) for a route-B checkpoint or VIP-Seg's released model."""
    import eval as eval_cli

    from models.vip_stage import CrossFormModule, VIPStage

    args = SimpleNamespace(checkpoint=ck.path, model="vipseg" if ck.kind == "vipseg" else "cascadeproto",
                           cvfold=1, checkpoint_cvfold=ck.fold, allow_seen_classes=False)
    model = eval_cli.load_model(args, device)
    protocol = eval_cli.protocol_check(args, model.checkpoint_args)  # raises on seen classes [D-22]
    if ck.kind == "vipseg":
        vm = model.model.vip_module
        for i in range(len(vm)):
            vm[i] = CrossFormModule(vm[i])  # this loaded instance only
        switches, attr = list(vm), "form"
        rule = r2.ReleasedRule(model)  # its hooks sit on the wrappers, which VIP-Seg's loop calls
    else:
        cfg = model.config
        if cfg.stage_type != "vip" or model.neck is not None or not all(isinstance(s, VIPStage) for s in model.stages):
            raise ValueError(f"{ck.name}: C4 reads route B's head trained with VIP-Seg's cross-term [DECISION D-36]")
        switches, attr = list(model.stages), "cross_form"
        rule = r2.OursRule(model)
    model.eval()  # after wrapping, so the wrappers are in evaluation mode too

    def set_form(form: str) -> None:
        for m in switches:
            setattr(m, attr, form)

    return rule, protocol, set_form


def check_c3(result: Dict, c3: Dict, name: str) -> Dict[str, float]:
    """C4.0b: the native passes must reproduce C3's mIoU of the same checkpoint."""
    ref = c3.get("checkpoints", {}).get(name)
    if ref is None:
        raise RuntimeError(f"C4.0b: {name} is not in {C3_JSON}; C3 must be run with the same names first")
    diffs = {"stored": 100.0 * abs(result["miou"]["native_stored"] - ref["miou"]["model"]),
             "swapped": 100.0 * abs(result["miou"]["native_swapped"] - ref["miou"]["model_swapped"])}
    if max(diffs.values()) > C3_TOL:
        raise RuntimeError(f"C4.0b: {name} native passes differ from C3 by {diffs} mIoU points")
    return diffs


def decide(out: Dict) -> Tuple[List[Tuple[str, str]], bool]:
    """Rule C4.1 and the C4.2 report; the flag is True when C4.1 holds on every checkpoint."""
    verdicts, ok = [], bool(out["checkpoints"])
    for name, r in out["checkpoints"].items():
        m = {k: 100.0 * v for k, v in r["miou"].items()}
        gap = abs(m["clean_stored"] - m["clean_swapped"])
        relabel = relabel_shift(r, "clean")
        holds = gap <= SWAP_GAP and abs(relabel) <= RELABEL
        ok = ok and holds
        verdicts.append((f"C4.1 {name}", f"clean stored {m['clean_stored']:.2f}, swapped {m['clean_swapped']:.2f} "
                                         f"(gap {gap:.3f}), relabel shift {relabel:+.4f}: "
                                         + ("sole carrier" if holds else "SECOND CARRIER, D-37 stops")))
        verdicts.append((f"C4.2 {name}", f"native {m['native_stored']:.2f} / swapped {m['native_swapped']:.2f} "
                                         f"(relabel shift {relabel_shift(r, 'native'):+.3f}); "
                                         f"clean {m['clean_stored']:.2f}; scrambled {m['scrambled_stored']:.2f}; "
                                         f"max relative logit difference scrambled vs native {r['form_error']:.2e}"))
    return verdicts, ok


@torch.no_grad()
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["run", "decide"])
    p.add_argument("--data_path")
    p.add_argument("--checkpoint", action="append", default=[])
    p.add_argument("--max_episodes", type=int, default=None, help="smoke runs only; skips C4.0b")
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)
    if args.stage == "decide":
        with open(args.out) as f:
            verdicts, ok = decide(json.load(f))
        for rule, text in verdicts:
            print(f"{rule:16s} {text}")
        return 0 if ok else 3
    from pipeline.episodes import make_episode, read_class_names

    dev = torch.device("cuda")
    names = read_class_names(args.data_path, "s3dis")
    ds, test_classes = r2.episodes_of("fixed100", args.data_path, 1)
    n_ep = len(ds) if args.max_episodes is None else min(args.max_episodes, len(ds))
    c3 = None
    if args.max_episodes is None:
        with open(os.path.join(REPO, C3_JSON)) as f:
            c3 = json.load(f)
    out = {"episodes": n_ep, "checkpoints": {}}
    for spec in args.checkpoint:
        ck = p0.parse_checkpoint(spec)
        if ck.fold != 1:
            raise ValueError(f"{ck.name}: C4 runs on S1 only [DECISION D-22]")
        rule, protocol, set_form = load_with_forms(ck, dev)
        counts = {pass_name(f, s): [] for f, s in PASSES}
        own = {pass_name(f, s): dict(points=0, own=0, other=0, background=0) for f, s in PASSES}
        form_error = 0.0
        try:
            for i in range(n_ep):
                item = ds[i]
                qy = item[3]  # [2, P], stored order
                native = None
                for form, swap in PASSES:
                    set_form(form)
                    e = make_episode(swapped(item) if swap else item, names).to(dev)
                    f_q, m_eff, _, logits = rule(e)
                    p0.check_identity(f_q, m_eff, logits)
                    if (form, swap) == ("native", False):
                        native = logits
                    elif (form, swap) == ("scrambled", False):  # C4.0a
                        err = (logits - native).abs().max().item() / max(1.0, native.abs().max().item())
                        form_error = max(form_error, err)
                        if err > FORM_RTOL:
                            raise RuntimeError(f"C4.0a: {ck.name} episode {i}: scrambled differs from native by {err:.2e}")
                    pred = logits.argmax(-1).cpu().numpy()  # [2, P] in the order fed
                    if swap:
                        pred = pred[::-1]  # back to the stored order
                    counts[pass_name(form, swap)].append(p0.episode_counts(pred, qy, item[4], test_classes))
                    own_update(own[pass_name(form, swap)], pred, qy, swap)
        finally:
            set_form("native")
            rule.close()
        stacked = {k: np.stack(v) for k, v in counts.items()}
        res = {"protocol": protocol, "form_error": form_error,
               "miou": {k: float(p0.miou_from_counts(v.sum(0))) for k, v in stacked.items()},
               "class_iou": {k: p0.class_ious(v.sum(0)).tolist() for k, v in stacked.items()},
               "own_class_share": {k: shares(v) for k, v in own.items()},
               "own_points": own["native_stored"]["points"],
               "paired": {"clean_swapped_vs_clean_stored": p0.paired_bootstrap(stacked["clean_stored"],
                                                                                stacked["clean_swapped"]),
                          "clean_stored_vs_native_stored": p0.paired_bootstrap(stacked["native_stored"],
                                                                                stacked["clean_stored"])}}
        if c3 is not None:
            res["c3_difference"] = check_c3(res, c3, ck.name)  # C4.0b
        out["checkpoints"][ck.name] = res
        m = res["miou"]
        print(f"[c4] {ck.name}: " + " | ".join(f"{k} {100 * v:.2f}" for k, v in m.items())
              + f" | relabel shift native {relabel_shift(res, 'native'):+.4f}, clean {relabel_shift(res, 'clean'):+.4f}"
              + f" | scrambled vs native {form_error:.2e}", flush=True)
    path = args.out if args.max_episodes is None else args.out.replace(".json", "_smoke.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    verdicts, ok = decide(out)
    for rule_name, text in verdicts:
        print(f"{rule_name:16s} {text}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
