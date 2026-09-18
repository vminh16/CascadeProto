"""CascadeProto end-to-end model behind the episode contract of pipeline/model_api.py.

Built in phases. Phase 10 implements the Table 4 "Baseline" row of [DECISION D-17]: shared VIP-Seg
features, point prototypes, and single-step matching `L = F^q P_pointᵀ` (Eq.23 with P = P_point).
The switches of spec 01 §3 are all present; configurations that need LMA (phase 11), the EPPM
cascade (phase 12) or ADRM (phase 13) raise NotImplementedError until those phases land.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prototypes import point_prototypes
from pipeline.episodes import Episode
from pipeline.model_api import EpisodeOutput

MODALITIES = ("text", "image", "audio")  # 03 §2
LOGIT_SCALES = ("none", "sqrt_D")  # [DECISION D-10]


@dataclass(frozen=True)
class CascadeProtoConfig:
    """Switches of spec 01 §3; the defaults are the full model."""

    use_lma: bool = True
    num_stages: int = 4
    use_gate: bool = True
    use_adrm: bool = True
    modality: str = "text"
    logit_scale: str = "none"
    l2norm_point_proto: bool = False

    def __post_init__(self):
        if not 0 <= self.num_stages <= 6:
            raise ValueError(f"num_stages must be in 0..6 (01 §3), got {self.num_stages}")
        if self.modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {self.modality!r}")
        if self.logit_scale not in LOGIT_SCALES:
            raise ValueError(f"logit_scale must be one of {LOGIT_SCALES}, got {self.logit_scale!r}")

    def check_implemented(self) -> None:
        if self.use_lma:
            raise NotImplementedError("use_lma=true needs the LMA and GMMN of phase 11")
        if self.num_stages > 0:
            raise NotImplementedError("num_stages > 0 needs the EPPM cascade of phase 12 (and ADRM, phase 13)")

    def to_dict(self) -> dict:
        return asdict(self)


class CascadeProto(nn.Module):
    def __init__(self, config: CascadeProtoConfig, feature_extractor: Optional[nn.Module] = None):
        super().__init__()
        config.check_implemented()
        self.config = config
        if feature_extractor is None:
            from models.vipseg_backbone import PointFeatureExtractor

            feature_extractor = PointFeatureExtractor()
        self.features = feature_extractor

    def forward(self, episode: Episode) -> EpisodeOutput:
        f_s, f_q = self.features.encode_episode(episode.support_x, episode.query_x)  # [N,K,P,D], [B_q,P,D]
        prototypes = point_prototypes(f_s, episode.support_y)  # [N+1, D] (Eq.3)
        if self.config.l2norm_point_proto:  # ablation only [DECISION D-10]
            prototypes = F.normalize(prototypes, dim=-1)  # [N+1, D]
        logits = torch.einsum("bpd,cd->bpc", f_q, prototypes)  # [B_q, P, N+1] (Eq.23, no temperature)
        if self.config.logit_scale == "sqrt_D":  # ablation only [DECISION D-10]
            logits = logits / math.sqrt(f_q.shape[-1])
        return EpisodeOutput(logits=logits, loss_gmmn=logits.new_zeros(()))  # no LMA -> no L_GMMN (D-17)
