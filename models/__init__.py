"""CascadeProto models package."""

from models.vipseg_backbone import VIPSegBackbone, extract_point_prototypes, GatingNetwork
from models.lma import (
    TextModalityAdapter,
    GMMNGenerator,
    LearnableModalityAdapter,
    fuse_initial_prototypes,
    format_category_prompt,
    generate_clip_text_embeddings
)
from models.eppm import (
    compute_shannon_entropy,
    InformationTheoreticGating,
    CrossAttentionRefinement,
    PrototypeDiffusion,
    AdaptiveFusionRecalibration,
    EPPMStage,
    EPPMCascade
)

__all__ = [
    "VIPSegBackbone",
    "extract_point_prototypes",
    "GatingNetwork",
    "TextModalityAdapter",
    "GMMNGenerator",
    "LearnableModalityAdapter",
    "fuse_initial_prototypes",
    "format_category_prompt",
    "generate_clip_text_embeddings",
    "compute_shannon_entropy",
    "InformationTheoreticGating",
    "CrossAttentionRefinement",
    "PrototypeDiffusion",
    "AdaptiveFusionRecalibration",
    "EPPMStage",
    "EPPMCascade",
]
