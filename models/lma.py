"""
Learnable Modality Adapters (LMA) and GMMN Generator for CascadeProto.
Implements:
1. Prompt Template generation for category descriptions (foreground & background).
2. Frozen CLIP feature extraction helper (Text modality).
3. TextModalityAdapter (2-layer MLP projection 512 -> 128).
4. GMMNGenerator (3-layer MLP distribution generator 256 -> 128).
5. LearnableModalityAdapter (end-to-end adapter + generator).
6. Initial Prototype Fusion: P_0 = P_point + P_modal.

Reference:
- 01_ARCHITECTURE_SPEC.md (Section 2, Module C)
- 03_MULTIMODAL_SPEC.md (Sections 1, 2, 3, 4, 5)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import clip
except ImportError:
    clip = None



def format_category_prompt(class_name: str, is_background: bool = False) -> str:
    """
    Format standard category prompt according to Section 1.1 of 03_MULTIMODAL_SPEC.md:
    Foreground: "This point cloud represents the {class_name}."
    Background: "This point cloud represents the background clutter."
    """
    if is_background or class_name.lower() in ("background", "clutter", "bg"):
        return "This point cloud represents the background clutter."
    return f"This point cloud represents the {class_name}."


def generate_clip_text_embeddings(
    class_names: List[str],
    clip_model=None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract normalized CLIP text embeddings for background and N foreground classes.
    Args:
        class_names: List of N foreground class names (e.g. ['ceiling', 'floor']).
        clip_model: Pre-loaded OpenAI CLIP model. If None, loaded automatically.
        device: Target device (CPU or CUDA).
    Returns:
        E_clip: Normalized CLIP embeddings of shape [1, N + 1, 512].
                Index 0 corresponds to background clutter.
    """
    if clip is None:
        raise ImportError("OpenAI CLIP is required. Please install via 'pip install git+https://github.com/openai/CLIP.git'")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if clip_model is None:
        import os
        model_name = "ViT-B/16" if os.path.exists(os.path.expanduser("~/.cache/clip/ViT-B-16.pt")) else "ViT-B/32"
        clip_model, _ = clip.load(model_name, device=device)
        clip_model.eval()


    # Construct (N+1) prompts: index 0 is background, indices 1..N are target classes
    prompts = [format_category_prompt("clutter", is_background=True)]
    for name in class_names:
        prompts.append(format_category_prompt(name, is_background=False))

    # Tokenize prompts
    text_tokens = clip.tokenize(prompts).to(device)

    # Encode with frozen CLIP text encoder
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).float()
        # L2 Normalize as per CLIP specification
        text_features = text_features / torch.clamp(text_features.norm(dim=-1, keepdim=True), min=1e-8)

    # Shape: [1, N + 1, 512]
    return text_features.unsqueeze(0)


class TextModalityAdapter(nn.Module):
    """
    2-layer Multi-Layer Perceptron (MLP) Adapter mapping 2D representation spaces (512-dim)
    to 3D point cloud space (128-dim).
    Formula: Adapter(E) = W_2 * ReLU(LayerNorm(W_1 * E + b_1)) + b_2
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 128, dropout_p: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, clip_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embeddings: [B, N+1, 512] normalized CLIP representations.
        Returns:
            E_adapted: [B, N+1, 128] adapted semantic representations.
        """
        x = self.fc1(clip_embeddings)   # [B, N+1, 128]
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                  # [B, N+1, 128]
        return x


class GMMNGenerator(nn.Module):
    """
    Generative Distribution Matcher (G) mapping adapted semantic representations
    concatenated with Gaussian noise z ~ N(0, I_D) to synthesized modal prototypes P_modal.
    Layers:
      Layer 1: Linear 256 -> 128, ReLU
      Layer 2: Linear 128 -> 128, ReLU
      Layer 3: Linear 128 -> 128 (Identity output)
    """
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.generator = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),  # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),      # 128 -> 128
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),      # 128 -> 128 (linear output)
        )

    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_embeddings: [B, N+1, 256] channel-concatenated [E_adapted; z].
        Returns:
            P_modal: [B, N+1, 128] synthesized modal prototypes.
        """
        return self.generator(fused_embeddings)


class LearnableModalityAdapter(nn.Module):
    """
    Complete LMA Pipeline for single-modality adaptation and prototype generation.
    Dataflow:
      E_CLIP [B, N+1, 512] -> Adapter -> E_adapted [B, N+1, 128]
      [E_adapted; z] [B, N+1, 256] -> Generator G -> P_modal [B, N+1, 128]
    """
    def __init__(self, clip_dim: int = 512, feat_dim: int = 128, dropout_p: float = 0.1):
        super().__init__()
        self.adapter = TextModalityAdapter(in_dim=clip_dim, out_dim=feat_dim, dropout_p=dropout_p)
        self.generator = GMMNGenerator(feat_dim=feat_dim)
        self.feat_dim = feat_dim

    def forward(
        self,
        clip_embeddings: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            clip_embeddings: [B, N+1, 512] normalized CLIP representations.
            noise: Optional [B, N+1, 128] Gaussian noise vector z.
                   If None, sampled from N(0, I_D) independently.
        Returns:
            P_modal: [B, N+1, 128] synthesized modal prototypes.
        """
        # 1. Project semantic embeddings to geometric feature dimension D = 128
        e_adapted = self.adapter(clip_embeddings)  # [B, N+1, 128]

        # 2. Sample standard normal Gaussian noise z ~ N(0, I_D)
        if noise is None:
            noise = torch.randn_like(e_adapted)

        # 3. Channel concatenation [E_adapted; z]: [B, N+1, 256]
        e_fused = torch.cat([e_adapted, noise], dim=-1)

        # 4. Generate synthesized modal prototypes P_modal: [B, N+1, 128]
        p_modal = self.generator(e_fused)
        return p_modal


def fuse_initial_prototypes(p_point: torch.Tensor, p_modal: torch.Tensor) -> torch.Tensor:
    """
    Initial Prototype Assembly via element-wise residual addition:
        P_0 = P_point + P_modal in R^[B, N+1, 128]
    Reference: Section 5 of 03_MULTIMODAL_SPEC.md.
    """
    assert p_point.shape == p_modal.shape, (
        f"Shape mismatch in prototype fusion: p_point {p_point.shape} vs p_modal {p_modal.shape}"
    )
    return p_point + p_modal
