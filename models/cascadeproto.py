"""CascadeProto end-to-end model behind the episode contract of pipeline/model_api.py.

All rows of Table 4 [DECISION D-17]:
* Baseline (`use_lma=false, num_stages=0`): `L = F^q P_pointᵀ` (Eq.23 with P = P_point).
* + LMA (`use_lma=true, num_stages=0`): `L = F^q (P^0)ᵀ`, `P^0 = P_point + P_modal` (Eq.9), plus L_GMMN.
* + Entropy Gate (`num_stages=1`) and + Cascade (`num_stages=T, use_adrm=false`): P^0 -> EPPM_1 -> ...
  -> P^T (Eq.22), prediction `L^T = F^q (P^T)ᵀ` (Eq.23). With T = 1, ADRM weighs a single stage by 1, so
  `use_adrm` does not change the prediction and no W_g is built.
* + ADRM, the full model (defaults): `L_final = Σ_t w_gate^(t) L^t` (Eq.24-25).
The switches of spec 01 §3 are all present; unimplemented ablation flag values raise NotImplementedError.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.gmmn_loss import FG_MODES, gmmn_loss
from models.adrm import DynamicRouting
from models.clip_text import DEFAULT_CLIP_VARIANT, ClipTextEmbedding
from models.eppm import (CROSS_ATTN_NORMS, CROSS_ATTN_SCALES, FUSION_WEIGHTS, GATE_TARGETS, EPPMStage,
                         stage_logits)
from models.lma import EVAL_NOISE, LearnableModalityAdapter
from models.prototypes import point_prototypes
from pipeline.episodes import Episode
from pipeline.model_api import EpisodeOutput

MODALITIES = ("text", "image", "audio")  # 03 §2
LOGIT_SCALES = ("none", "sqrt_D")  # [DECISION D-10]
CROSS_ATTN = ("channel", "two_hop")  # [DECISION D-01]
DIFFUSION_INPUTS = ("post_relu", "pre_relu")  # [DECISION D-14]


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
    cross_attn: str = "channel"  # [DECISION D-01]
    cross_attn_scale: str = "sqrt_d"  # [DECISION D-01]
    cross_attn_norm: str = "none"  # [DECISION D-18]
    gate_target: str = "prototype"  # [DECISION D-02]
    fusion_weight: str = "per_query"  # [DECISION D-11]
    diffusion_input: str = "post_relu"  # [DECISION D-14]

    def __post_init__(self):
        if not 0 <= self.num_stages <= 6:
            raise ValueError(f"num_stages must be in 0..6 (01 §3), got {self.num_stages}")
        for name, value, allowed in (("modality", self.modality, MODALITIES),
                                     ("logit_scale", self.logit_scale, LOGIT_SCALES),
                                     ("eval_noise", self.eval_noise, EVAL_NOISE),
                                     ("gmmn_fg_mode", self.gmmn_fg_mode, FG_MODES),
                                     ("cross_attn", self.cross_attn, CROSS_ATTN),
                                     ("cross_attn_scale", self.cross_attn_scale, CROSS_ATTN_SCALES),
                                     ("cross_attn_norm", self.cross_attn_norm, CROSS_ATTN_NORMS),
                                     ("gate_target", self.gate_target, GATE_TARGETS),
                                     ("fusion_weight", self.fusion_weight, FUSION_WEIGHTS),
                                     ("diffusion_input", self.diffusion_input, DIFFUSION_INPUTS)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

    def check_implemented(self) -> None:
        if self.use_lma and self.modality != "text":
            raise NotImplementedError(f"modality {self.modality!r} is not implemented yet (03 §2.2)")
        for name, value, default in (("cross_attn", self.cross_attn, "channel"),
                                     ("diffusion_input", self.diffusion_input, "post_relu")):
            if value != default:
                raise NotImplementedError(f"{name}={value!r} is not implemented (00 D-01, D-14)")

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
        # T stages with their own parameters [PAPER §3.5] [DECISION D-16]
        self.stages = nn.ModuleList(
            EPPMStage(use_gate=config.use_gate, cross_attn_scale=config.cross_attn_scale,
                      fusion_weight=config.fusion_weight, cross_attn_norm=config.cross_attn_norm,
                      gate_target=config.gate_target)
            for _ in range(config.num_stages))
        # ADRM over T >= 2 stages; with T = 1 its weight is 1 and W_g could not learn [DECISION D-17]
        self.routing = DynamicRouting(config.num_stages) if config.use_adrm and config.num_stages >= 2 else None

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
        if len(self.stages) == 0:
            logits = torch.einsum("bpd,cd->bpc", f_q, prototypes)  # [B_q, P, N+1] (Eq.23, no temperature)
        else:
            p = prototypes.unsqueeze(0).expand(f_q.shape[0], -1, -1)  # P^0 per query [B_q, N+1, D] (02 §4.5)
            all_logits = []
            for stage in self.stages:
                p = stage(p, f_s, f_q)  # P^t (Eq.22)
                all_logits.append(stage_logits(f_q, p))  # L^t [B_q, P, N+1] (Eq.23)
            if self.routing is not None:
                logits = self.routing(all_logits, f_q)  # L_final (Eq.24-25)
            else:
                logits = all_logits[-1]  # L^T without ADRM [DECISION D-17]
        if self.config.logit_scale == "sqrt_D":  # ablation only [DECISION D-10]
            logits = logits / math.sqrt(f_q.shape[-1])
        return EpisodeOutput(logits=logits, loss_gmmn=loss_gmmn)
