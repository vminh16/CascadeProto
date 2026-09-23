"""VIP-Seg's own PEM / PDM behind the cascade-stage contract [DECISION D-25]. **Beyond the paper.**

The reference head of phase 16: it makes VIP-Seg's prototype modules runnable inside CascadeProto, so
that "our stage against theirs" is one switch rather than two training scripts. The modules themselves
are imported from the inherited `models/vipseg.py` and never edited (AGENTS guardrail 2); this file only
adapts the call and reproduces VIP-Seg's alternation and outer residual
[VIPSEG models/vipseg.py:154-160].

Their internal reshape makes one query's prediction depend on the other queries of the episode
(research note §4.4); that is VIP-Seg's behaviour, kept here on purpose, and it is why this stage is a
reference to compare against, not a design to copy.
"""

from typing import Optional

import torch
import torch.nn as nn


def build_vip_module(step: int) -> nn.Module:
    """PEM on even steps, PDM on odd ones, as VIP-Seg alternates them [VIPSEG models/vipseg.py:154-160].

    Imported lazily: `models.vipseg` pulls in the encoder and `pointnet2_ops`, which only exist on the
    GPU environment (05 §2, gate G2).
    """
    from models.vipseg import PrototypeDifferenceModule, PrototypeEnhancementModule

    return PrototypeEnhancementModule() if step % 2 == 0 else PrototypeDifferenceModule()


class VIPStage(nn.Module):
    """One VIP-Seg reasoning step with the signature of `EPPMStage`.

    `module(query [B_q, 2048, D], supports [N, K, 2048, D], prototype [B_q, N+1, D]) -> [B_q, N+1, D]`
    is VIP-Seg's own call [VIPSEG models/vipseg.py:155-158]; odd steps add the outer residual VIP-Seg
    applies to the PDM output [VIPSEG models/vipseg.py:157]. `module` is injected in tests, where
    `models.vipseg` cannot be imported.
    """

    def __init__(self, step: int = 0, module: Optional[nn.Module] = None):
        super().__init__()
        self.step = step
        self.outer_residual = step % 2 == 1
        self.module = build_vip_module(step) if module is None else module

    def forward(self, p_prev: torch.Tensor, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """p_prev [B_q, N+1, D], f_s [N, K, 2048, D], f_q [B_q, 2048, D] -> P^t [B_q, N+1, D]."""
        if p_prev.dim() != 3 or p_prev.shape[0] != f_q.shape[0] or p_prev.shape[1] != f_s.shape[0] + 1:
            raise ValueError(f"P^(t-1) {tuple(p_prev.shape)} does not match F^s {tuple(f_s.shape)} and "
                             f"F^q {tuple(f_q.shape)}")
        out = self.module(f_q, f_s, p_prev)  # [B_q, N+1, D]
        return out + p_prev if self.outer_residual else out  # [VIPSEG models/vipseg.py:157]
