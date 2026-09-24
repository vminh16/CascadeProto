"""CascadeProto end-to-end model behind the episode contract of pipeline/model_api.py.

All rows of Table 4 [DECISION D-17]:
* Baseline (`use_lma=false, num_stages=0`): `L = F^q P_pointᵀ` (Eq.23 with P = P_point).
* + LMA (`use_lma=true, num_stages=0`): `L = F^q (P^0)ᵀ`, `P^0 = P_point + P_modal` (Eq.9), plus L_GMMN.
* + Entropy Gate (`num_stages=1`) and + Cascade (`num_stages=T, use_adrm=false`): P^0 -> EPPM_1 -> ...
  -> P^T (Eq.22), prediction `L^T = F^q (P^T)ᵀ` (Eq.23). With T = 1, ADRM weighs a single stage by 1, so
  `use_adrm` does not change the prediction and no W_g is built.
* + ADRM, the full model (defaults): `L_final = Σ_t w_gate^(t) L^t` (Eq.24-25).
The switches of spec 01 §3 are all present; unimplemented ablation flag values raise NotImplementedError.

Beyond the paper, `distill_beta > 0` adds the oracle-direction loss of [DECISION D-29] on the pairwise
decision functions of `L_final` in training mode; with 0 it is only logged, without gradient.
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
from models.eppm import (CROSS_ATTN_NORMS, CROSS_ATTN_SCALES, CROSS_ATTN_SUPPORTS, EQ19_SELF, FUSION_WEIGHTS,
                         GATE_TARGETS, EPPMStage, stage_logits)
from models.lma import EVAL_NOISE, LearnableModalityAdapter
from models.neck import NECKS, build_neck
from models.oracle_distill import oracle_distill_loss
from models.prototypes import point_prototypes
from pipeline.episodes import Episode
from pipeline.model_api import EpisodeOutput

MODALITIES = ("text", "image", "audio")  # 03 §2
STAGE_TYPES = ("eppm", "eppm_s", "vip")  # [DECISION D-24] [DECISION D-25], beyond the paper
# Switches that only mean something for the printed EPPM stage; a non-default value with another
# stage type would be silently ignored, so it raises instead (AGENTS guardrail 7).
EPPM_ONLY = ("use_gate", "gate_target", "eq19_self", "cross_attn_norm", "fusion_weight", "diffusion_input")
VIP_IGNORES = ("cross_attn_scale", "cross_attn_support")  # VIP-Seg's own modules fix both
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
    cross_attn_support: str = "class_slots"  # [DECISION D-23], beyond the paper's D-01 reading
    gate_target: str = "prototype"  # [DECISION D-02]
    eq19_self: str = "none"  # [DECISION D-19], beyond the paper
    stage_type: str = "eppm"  # [DECISION D-24] [DECISION D-25], beyond the paper
    fusion_weight: str = "per_query"  # [DECISION D-11]
    diffusion_input: str = "post_relu"  # [DECISION D-14]
    distill_beta: float = 0.0  # [DECISION D-29], beyond the paper; 0 = the paper's objective
    neck: str = "none"  # [DECISION D-33], beyond the paper
    neck_alpha_init: float = 0.0  # [DECISION D-34]; 0 keeps D-33's identity at initialisation

    def __post_init__(self):
        if not 0 <= self.num_stages <= 6:
            raise ValueError(f"num_stages must be in 0..6 (01 §3), got {self.num_stages}")
        if not math.isfinite(self.neck_alpha_init) or (self.neck_alpha_init != 0 and self.neck == "none"):
            raise ValueError(f"neck_alpha_init must be finite and needs a neck [DECISION D-34], got "
                             f"{self.neck_alpha_init} with neck={self.neck!r}")
        if not (math.isfinite(self.distill_beta) and self.distill_beta >= 0):
            raise ValueError(f"distill_beta must be a finite value >= 0 [DECISION D-29], got {self.distill_beta}")
        for name, value, allowed in (("modality", self.modality, MODALITIES),
                                     ("logit_scale", self.logit_scale, LOGIT_SCALES),
                                     ("eval_noise", self.eval_noise, EVAL_NOISE),
                                     ("gmmn_fg_mode", self.gmmn_fg_mode, FG_MODES),
                                     ("cross_attn", self.cross_attn, CROSS_ATTN),
                                     ("cross_attn_scale", self.cross_attn_scale, CROSS_ATTN_SCALES),
                                     ("cross_attn_norm", self.cross_attn_norm, CROSS_ATTN_NORMS),
                                     ("cross_attn_support", self.cross_attn_support, CROSS_ATTN_SUPPORTS),
                                     ("gate_target", self.gate_target, GATE_TARGETS),
                                     ("eq19_self", self.eq19_self, EQ19_SELF),
                                     ("stage_type", self.stage_type, STAGE_TYPES),
                                     ("fusion_weight", self.fusion_weight, FUSION_WEIGHTS),
                                     ("diffusion_input", self.diffusion_input, DIFFUSION_INPUTS),
                                     ("neck", self.neck, NECKS)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

    def check_implemented(self) -> None:
        if self.use_lma and self.modality != "text":
            raise NotImplementedError(f"modality {self.modality!r} is not implemented yet (03 §2.2)")
        if self.stage_type != "eppm":  # switches of the printed stage that another stage cannot honour
            defaults = CascadeProtoConfig()
            ignored = [f for f in EPPM_ONLY if getattr(self, f) != getattr(defaults, f)]
            ignored += [f for f in VIP_IGNORES if self.stage_type == "vip" and getattr(self, f) != getattr(defaults, f)]
            if ignored:
                raise ValueError(f"stage_type={self.stage_type!r} ignores {ignored}; leave them at their "
                                 f"defaults so that a run's configuration describes what it ran "
                                 f"[DECISION D-24] [DECISION D-25]")
        for name, value, default in (("cross_attn", self.cross_attn, "channel"),
                                     ("diffusion_input", self.diffusion_input, "post_relu")):
            if value != default:
                raise NotImplementedError(f"{name}={value!r} is not implemented (00 D-01, D-14)")

    def to_dict(self) -> dict:
        return asdict(self)


def build_stage(config: "CascadeProtoConfig", step: int) -> nn.Module:
    """Stage `step` of the cascade: the printed EPPM, EPPM-S [D-24] or VIP-Seg's own module [D-25].

    All three take `(P^{t-1} [B_q, N+1, D], F^s [N, K, 2048, D], F^q [B_q, 2048, D])` and return
    `P^t [B_q, N+1, D]`, so the cascade, ADRM and the losses are untouched by the choice.
    """
    if config.stage_type == "eppm":
        return EPPMStage(use_gate=config.use_gate, cross_attn_scale=config.cross_attn_scale,
                         fusion_weight=config.fusion_weight, cross_attn_norm=config.cross_attn_norm,
                         gate_target=config.gate_target, eq19_self=config.eq19_self,
                         cross_attn_support=config.cross_attn_support)
    if config.stage_type == "eppm_s":
        from models.eppm_s import EPPMSharedStage

        return EPPMSharedStage(cross_attn_scale=config.cross_attn_scale, support=config.cross_attn_support)
    from models.vip_stage import VIPStage  # imports models.vipseg, which needs the GPU environment

    return VIPStage(step)


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
        # Support -> query attention before the prototypes, identity at initialisation [DECISION D-33]
        self.neck = build_neck(config.neck, alpha_init=config.neck_alpha_init)
        if config.use_lma:
            self.lma = LearnableModalityAdapter(eval_noise=config.eval_noise)
            # Frozen CLIP stays outside the module tree: not in state_dict, untouched by .to()/.double() (03 §2.1)
            self.text = text_embedding if text_embedding is not None else ClipTextEmbedding(config.clip_variant)
        # T stages with their own parameters [PAPER §3.5] [DECISION D-16]
        self.stages = nn.ModuleList(build_stage(config, step) for step in range(config.num_stages))
        # ADRM over T >= 2 stages; with T = 1 its weight is 1 and W_g could not learn [DECISION D-17]
        self.routing = DynamicRouting(config.num_stages) if config.use_adrm and config.num_stages >= 2 else None

    def cascade(self, episode: Episode):
        """(F^q [B_q, P, D], P^0 [N+1, D], [P^1..P^T] each [B_q, N+1, D], L_GMMN) of one episode.

        The forward's own computation up to the stage prototypes; `experiments/r2_distill_eval.py` reads
        it for the diagnostics of [DECISION D-29].
        """
        f_s, f_q = self.features.encode_episode(episode.support_x, episode.query_x)  # [N,K,P,D], [B_q,P,D]
        if self.neck is not None:
            f_s = self.neck(f_s, f_q)  # [N, K, P, D], the prototypes and the head see F_s' [DECISION D-33]
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
        steps = []  # P^1..P^T
        p = prototypes.unsqueeze(0).expand(f_q.shape[0], -1, -1)  # P^0 per query [B_q, N+1, D] (02 §4.5)
        for stage in self.stages:
            p = stage(p, f_s, f_q)  # P^t [B_q, N+1, D] (Eq.22)
            steps.append(p)
        return f_q, prototypes, steps, loss_gmmn

    def forward(self, episode: Episode) -> EpisodeOutput:
        f_q, prototypes, steps, loss_gmmn = self.cascade(episode)  # [B_q,P,D], [N+1,D], T x [B_q,N+1,D]
        if not steps:
            logits = torch.einsum("bpd,cd->bpc", f_q, prototypes)  # [B_q, P, N+1] (Eq.23, no temperature)
        else:
            all_logits = [stage_logits(f_q, p) for p in steps]  # L^t [B_q, P, N+1] (Eq.23)
            if self.routing is not None:
                logits = self.routing(all_logits, f_q)  # L_final (Eq.24-25)
            else:
                logits = all_logits[-1]  # L^T without ADRM [DECISION D-17]
        if self.config.logit_scale == "sqrt_D":  # ablation only [DECISION D-10]
            logits = logits / math.sqrt(f_q.shape[-1])
        if not self.training:  # the query labels are never read at evaluation [DECISION D-29]
            return EpisodeOutput(logits=logits, loss_gmmn=loss_gmmn)
        beta = self.config.distill_beta
        if beta > 0:
            loss_distill = oracle_distill_loss(logits, f_q, episode.query_y)  # scalar (02 §14)
        else:  # logged for comparison, no gradient and no effect on training [DECISION D-29]
            with torch.no_grad():
                loss_distill = oracle_distill_loss(logits.detach(), f_q, episode.query_y)  # scalar
        return EpisodeOutput(logits=logits, loss_gmmn=loss_gmmn, loss_distill=loss_distill, distill_weight=beta)

    def effective_prototype(self, f_q: torch.Tensor, p0: torch.Tensor, steps) -> torch.Tensor:
        """M_eff [B_q, N+1, D] with `L_final = F^q M_effᵀ` (up to logit_scale), for diagnostics [DECISION D-29].

        f_q [B_q, P, D]; p0 = P^0 [N+1, D]; steps = [P^1..P^T], each [B_q, N+1, D]. ADRM weighs the stages
        (Eq.24-25), without ADRM the last stage is the prediction [DECISION D-17], without stages P^0 is.
        """
        if not steps:
            return p0.unsqueeze(0).expand(f_q.shape[0], -1, -1)  # [B_q, N+1, D]
        if self.routing is not None:
            w = self.routing.weights(f_q)  # [B_q, T]
            return torch.einsum("bt,tbcd->bcd", w, torch.stack(list(steps)))  # [B_q, N+1, D]
        return steps[-1]  # [B_q, N+1, D]
