"""PROT-1..6 (05 §3.12, gate G1): eval.py refuses to score classes a checkpoint was trained on [DECISION D-22].

S0's training classes are fold 1's test classes and vice versa [VIPSEG dataloaders/s3dis.py:20-31], so a
checkpoint scored with the other --cvfold, or one trained with --train_classes all [D-21], is scored on
classes it has seen (report §3.5-3.6). CPU only; no data, no model weights.
"""

import os

import pytest
import torch

import eval as eval_script

BASE = ["--dataset", "s3dis", "--data_path", "x", "--n_way", "2", "--k_shot", "1", "--checkpoint", "c.pt"]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def args(cvfold, *extra):
    return eval_script.parse_args(BASE + ["--cvfold", str(cvfold), *extra])


def trained(cvfold, train_classes="split"):
    return {"cvfold": cvfold, "train_classes": train_classes}  # the part of vars(args) that train.py stores


@pytest.mark.parametrize("fold", [0, 1])
def test_prot1_same_fold_is_clean(fold):
    assert eval_script.protocol_check(args(fold), trained(fold)) == "clean"


@pytest.mark.parametrize("train_fold,eval_fold", [(0, 1), (1, 0)])
def test_prot2_other_fold_is_refused_and_only_a_flagged_diagnostic(train_fold, eval_fold):
    with pytest.raises(ValueError, match="seen"):
        eval_script.protocol_check(args(eval_fold), trained(train_fold))
    status = eval_script.protocol_check(args(eval_fold, "--allow_seen_classes", "true"), trained(train_fold))
    assert status.startswith("SEEN-CLASS DIAGNOSTIC") and f"trained on S{train_fold}" in status


def test_prot3_training_on_all_classes_is_refused_even_on_the_same_fold():
    with pytest.raises(ValueError, match="train_classes all"):
        eval_script.protocol_check(args(0), trained(0, "all"))
    assert "D-21" in eval_script.protocol_check(args(0, "--allow_seen_classes", "true"), trained(0, "all"))


def test_prot4_unrecorded_fold_must_be_stated_and_must_not_contradict_the_checkpoint():
    with pytest.raises(ValueError, match="--checkpoint_cvfold"):
        eval_script.protocol_check(args(0), None)  # VIP-Seg's released checkpoints record no fold
    assert eval_script.protocol_check(args(0, "--checkpoint_cvfold", "0"), None) == "clean"
    with pytest.raises(ValueError, match="seen"):
        eval_script.protocol_check(args(1, "--checkpoint_cvfold", "0"), None)
    with pytest.raises(ValueError, match="contradicts"):
        eval_script.protocol_check(args(1, "--checkpoint_cvfold", "1"), trained(0))
    # Checkpoints written before train.py stored train_classes default to the split protocol
    assert eval_script.protocol_check(args(1), {"cvfold": 1}) == "clean"


def test_prot5_main_checks_before_any_episode_is_built(tmp_path, monkeypatch):
    class Stub(torch.nn.Module):
        checkpoint_args = trained(0)

    def no_dataset(*a, **k):
        raise AssertionError("episodes were built before the protocol check")

    monkeypatch.setattr(eval_script, "load_model", lambda a, d: Stub())
    monkeypatch.setattr(eval_script, "build_eval_dataset", no_dataset)
    monkeypatch.setattr(eval_script, "read_class_names", lambda *a: [])
    with pytest.raises(ValueError, match="refusing"):
        eval_script.main(BASE + ["--cvfold", "1", "--save_dir", str(tmp_path)])


def test_prot6_existing_seen_class_scripts_declare_themselves_diagnostics():
    def code(name):  # the script without its comment lines
        with open(os.path.join(REPO, "experiments", name)) as f:
            return "".join(line for line in f if not line.lstrip().startswith("#"))

    seen = code("run_seen.sh")
    assert seen.count("--allow_seen_classes true") == 1 and "--checkpoint_cvfold 0" in seen
    rescore = code("rescore_alt_metrics.sh")
    assert "--allow_seen_classes" not in rescore and "--checkpoint_cvfold 0" in rescore
