"""Gate G0: environment checks (05 §3.1).

ENV-1  required packages import
ENV-2  CUDA is available, torch has kernels for this GPU, and the extensions run on it
ENV-3  inherited files are byte-identical to the pinned VIP-Seg commit
ENV-4  models/encoder.py imports mamba_ssm unconditionally and has no fallback block

Run on the target Linux/WSL2 machine: `pytest tests/test_environment.py -v`.
ENV-1 and ENV-2 carry the `cuda` marker because they need the GPU extensions; ENV-3 and ENV-4
also run in the CPU gate G1.
"""

import ast
import hashlib
import importlib
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]

# Git blob SHA-1 of each file at VIP-Seg commit 28aedc5093c0d386d526864c49505ae6921b1600
# (GitHub API: git/trees/<commit>?recursive=1). Hashing as a git blob lets anyone re-check
# these values against the upstream tree.
PINNED_BLOBS = {
    "dataloaders/loader.py": "8284a06632",
    "dataloaders/s3dis.py": "72440eee1a",
    "dataloaders/scannet.py": "c45308d8f0",
    "preprocess/collect_s3dis_data.py": "61c0f0bf81",
    "preprocess/collect_scannet_data.py": "b3ecb96458",
    "preprocess/room2blocks.py": "42873ab54e",
    "utils/checkpoint_util.py": "4f9b8db9ab",
    "utils/cuda_util.py": "d79e2b6bce",
    "utils/logger.py": "57a4a46ddf",
    "models/encoder.py": "364c75bd43",
    "models/mamba_block.py": "5d01aedaed",
    "models/model_utils.py": "39804f3845",
    "models/vipseg.py": "86920697f3",
    "models/vipseg_learner.py": "acd24b2150",
    "runs/training_free.py": "c710a180f4",
    "runs/training.py": "c15f32159d",
    "runs/evaluate.py": "798cf3c709",
    "main.py": "2bbdd037dc",
}


def git_blob_sha1(path: Path) -> str:
    """Git blob SHA-1 of the file with CRLF normalised to LF (Windows checkouts use CRLF)."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@pytest.mark.cuda
@pytest.mark.parametrize(
    "module",
    [
        "mamba_ssm",
        "mamba_ssm.modules.mamba_simple",
        "pointnet2_ops_lib.pointnet2_ops.pointnet2_utils",
        "clip",
        "h5py",
        "transforms3d",
        "timm",
    ],
)
def test_env1_required_packages_import(module):
    importlib.import_module(module)


@pytest.mark.cuda
def test_env2_cuda_runs_extensions_on_this_gpu():
    assert torch.cuda.is_available(), "CUDA is not available"
    major, minor = torch.cuda.get_device_capability(0)
    # A cubin built for sm_XY runs on any GPU of the same major version with minor >= Y
    # (e.g. sm_86 kernels on an L4, sm_89); PTX (compute_XY) is JIT-compiled for any newer GPU.
    runnable = [
        a for a in torch.cuda.get_arch_list()
        if (a.startswith("sm_") and int(a[3:-1]) == major and int(a[-1]) <= minor)
        or (a.startswith("compute_") and (int(a[8:-1]), int(a[-1])) <= (major, minor))
    ]
    assert runnable, (
        f"torch {torch.__version__} has no kernels for {torch.cuda.get_device_name(0)} "
        f"(sm_{major}{minor}); built for {torch.cuda.get_arch_list()}"
    )

    from mamba_ssm.modules.mamba_simple import Mamba
    from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

    xyz = torch.rand(1, 64, 3, device="cuda")  # [B, N, 3]
    idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), 8)  # [B, 8]
    assert idx.shape == (1, 8)
    assert len(set(idx[0].tolist())) == 8

    mixer = Mamba(d_model=16).cuda()
    out = mixer(torch.randn(1, 32, 16, device="cuda"))  # [B, L, d_model]
    assert out.shape == (1, 32, 16)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("rel_path", sorted(PINNED_BLOBS))
def test_env3_inherited_files_match_pinned_commit(rel_path):
    sha = git_blob_sha1(REPO / rel_path)
    assert sha.startswith(PINNED_BLOBS[rel_path]), (
        f"{rel_path} differs from VIP-Seg 28aedc5 (blob {sha[:10]}, expected {PINNED_BLOBS[rel_path]}); "
        "inherited files are read-only (AGENTS.md guardrail 2)"
    )


def test_env4_encoder_has_no_fallback():
    tree = ast.parse((REPO / "models/encoder.py").read_text(encoding="utf-8"))

    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "mamba_ssm.modules.mamba_simple" in top_level_imports, "mamba_ssm must be imported unconditionally"
    assert "pointnet2_ops_lib.pointnet2_ops" in top_level_imports, "pointnet2_ops must be imported unconditionally"

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert "TransBlock" not in names, "encoder must not fall back to TransBlock"
    assert "furthest_point_sample_py" not in names, "encoder must not fall back to a Python FPS"
