"""CascadeProto end-to-end model behind the episode contract of pipeline/model_api.py.

Built in phases. Implemented so far, as rows of Table 4 [DECISION D-17]:
* Baseline (`use_lma=false, num_stages=0`): `L = F^q P_pointᵀ` (Eq.23 with P = P_point).
* + LMA (`use_lma=true, num_stages=0`): `L = F^q (P^0)ᵀ`, `P^0 = P_point + P_modal` (Eq.9), plus L_GMMN.
The switches of spec 01 §3 are all present; configurations that need the EPPM cascade (phase 12) or
ADRM (phase 13) raise NotImplementedError until those phases land.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.gmmn_loss import FG_MODES, gmmn_loss
from models.clip_text import DEFAULT_CLIP_VARIANT, ClipTextEmbedding
from models.lma import EVAL_NOISE, LearnableModalityAdapter
from models.prototypes import point_prototypes
from pipeline.episodes import Episode
from pipeline.model_api import EpisodeOutput

MODALITIES = ("text", "image", "audio")  # 03 §2
LOGIT_SCALES = ("none", "sqrt_D")  # [DECISION D-10]


@dataclass(frozen=True)
class CascadeProtoConfig:
    """Switches of spec 01 §3; the defaults are the full model.

    Checkpoints store `to_dict()`; fields added in later phases have defaults, so older checkpoints
    still rebuild the architecture they were trained with.
    """

    use_lma: bool = True
    num_stages: int = 4
    use_gate: bool = True
    use_adrm: bool = True
    modality: str = "text"
    logit_scale: str = "none"
    l2norm_point_proto: bool = False
    clip_variant: str = DEFAULT_CLIP_VARIANT  # [DECISION D-13]
    eval_noise: str = "zero"  # [DECISION D-06]
    gmmn_fg_mode: str = "joint"  # [DECISION D-04]
    gmmn_detach_point: bool = False  # [DECISION D-04]

    def __post_init__(self):
        if not 0 <= self.num_stages <= 6:
            raise ValueError(f"num_stages must be in 0..6 (01 §3), got {self.num_stages}")
        for name, value, allowed in (("modality", self.modality, MODALITIES),
                                     ("logit_scale", self.logit_scale, LOGIT_SCALES),
                                     ("eval_noise", self.eval_noise, EVAL_NOISE),
                                     ("gmmn_fg_mode", self.gmmn_fg_mode, FG_MODES)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

    def check_implemented(self) -> None:
        if self.use_lma and self.modality != "text":
            raise NotImplementedError(f"modality {self.modality!r} is not implemented yet (03 §2.2)")
        if self.num_stages > 0:
            raise NotImplementedError("num_stages > 0 needs the EPPM cascade of phase 12 (and ADRM, phase 13)")

    def to_dict(self) -> dict:
        return asdict(self)


class CascadeProto(nn.Module):
    def __init__(self, config: CascadeProtoConfig, feature_extractor: Optional[nn.Module] = None,
                 text_embedding: Optional[ClipTextEmbedding] = None):
        super().__init__()
        config.check_implemented()
        self.config = config
        if feature_extractor is None:
            from models.vipseg_backbone import PointFeatureExtractor

            feature_extractor = PointFeatureExtractor()
        self.features = feature_extractor
        if config.use_lma:
            self.lma = LearnableModalityAdapter(eval_noise=config.eval_noise)
            # Frozen CLIP stays outside the module tree: not in state_dict, untouched by .to()/.double() (03 §2.1)
            self.text = text_embedding if text_embedding is not None else ClipTextEmbedding(config.clip_variant)

    def forward(self, episode: Episode) -> EpisodeOutput:
        f_s, f_q = self.features.encode_episode(episode.support_x, episode.query_x)  # [N,K,P,D], [B_q,P,D]
        p_point = point_prototypes(f_s, episode.support_y)  # [N+1, D] (Eq.3)
        if self.config.l2norm_point_proto:  # ablation only [DECISION D-10]
            p_point = F.normalize(p_point, dim=-1)  # [N+1, D]
        if self.config.use_lma:
            e_clip = self.text(episode.class_names, f_q.device).to(f_q.dtype)  # [N+1, 512]
            p_modal = self.lma(e_clip)  # [N+1, D] (Eq.4-6)
            loss_gmmn = gmmn_loss(p_modal, p_point, fg_mode=self.config.gmmn_fg_mode,
                                  detach_point=self.config.gmmn_detach_point)  # scalar (Eq.7-8)
            prototypes = p_point + p_modal  # P^0 [N+1, D] (Eq.9)
        else:
            loss_gmmn = f_q.new_zeros(())  # no LMA -> no L_GMMN [DECISION D-17]
            prototypes = p_point  # [N+1, D]
        logits = torch.einsum("bpd,cd->bpc", f_q, prototypes)  # [B_q, P, N+1] (Eq.23, no temperature)
        if self.config.logit_scale == "sqrt_D":  # ablation only [DECISION D-10]
            logits = logits / math.sqrt(f_q.shape[-1])
        return EpisodeOutput(logits=logits, loss_gmmn=loss_gmmn)
