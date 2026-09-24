"""VIP-Seg's own PEM / PDM behind the cascade-stage contract [DECISION D-25]. **Beyond the paper.**

The reference head of phase 16: it makes VIP-Seg's prototype modules runnable inside CascadeProto, so
that "our stage against theirs" is one switch rather than two training scripts. The modules themselves
are imported from the inherited `models/vipseg.py` and never edited (AGENTS guardrail 2); this file only
adapts the call and reproduces VIP-Seg's alternation and outer residual
[VIPSEG models/vipseg.py:154-160].

Their internal reshape makes one query's prediction depend on the other queries of the episode and on
the query's position in it (research note §4.4, D-35 outcome); that is VIP-Seg's behaviour, kept by
`cross_form="native"`. [DECISION D-36] adds `vip_module_forward`, the same PEM/PDM computation written out
from the inherited source and reading the inherited module's own weights, with the cross-term either as
VIP-Seg computes it (`scrambled`, which must reproduce `native`) or per (query, slot) (`clean`), which makes
the head equivariant in the query index [DECISION D-37].
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CROSS_FORMS = ("native", "scrambled", "clean")  # [DECISION D-36]
SCALE = 128.0 ** 0.5  # VIP-Seg's √128 in both self- and cross-correlation [VIPSEG models/vipseg.py:269,288]
MODULE_NAMES = ("PrototypeEnhancementModule", "PrototypeDifferenceModule")


def build_vip_module(step: int) -> nn.Module:
    """PEM on even steps, PDM on odd ones, as VIP-Seg alternates them [VIPSEG models/vipseg.py:154-160].

    Imported lazily: `models.vipseg` pulls in the encoder and `pointnet2_ops`, which only exist on the
    GPU environment (05 §2, gate G2).
    """
    from models.vipseg import PrototypeDifferenceModule, PrototypeEnhancementModule

    return PrototypeEnhancementModule() if step % 2 == 0 else PrototypeDifferenceModule()


def cross_attention(que: torch.Tensor, sup: torch.Tensor, form: str) -> torch.Tensor:
    """Softmax cross-correlation A [B_q, N+1, D, D] from Q' [B_q, 72, D] and S' [N+1, 72, D] [DECISION D-36].

    `scrambled` is VIP-Seg's computation [VIPSEG models/vipseg.py:285-291]: `reshape(72, -1)` reads the rows
    of all queries and all slots as one matrix, so entry (b, w) sums Q'[r // (72/B_q), B_q (r mod 72/B_q) + b]
    against S'[r // (72/(N+1)), (N+1)(r mod 72/(N+1)) + w] over r (D-35 point 5). The reshape across the
    query and slot axes breaks AGENTS guardrail 6 on purpose: it is the behaviour under test.
    `clean` is one attention per query b and slot w, `softmax(Q'_bᵀ S'_w / √128)`, D-01's form with VIP-Seg's
    scale.
    """
    b, proj, d = que.shape
    way = sup.shape[0]
    if form == "scrambled":
        q2, s2 = que.reshape(proj, -1), sup.reshape(proj, -1)  # [72, B_q*D], [72, (N+1)*D], VIP-Seg's own
        a = torch.matmul(q2.transpose(0, 1) / SCALE, s2)  # [B_q*D, (N+1)*D]
        a = a.reshape(b, d, way, d).permute(0, 2, 1, 3)  # [B_q, N+1, D, D]
    elif form == "clean":
        a = torch.einsum("bcd,wce->bwde", que, sup) / SCALE  # [B_q, N+1, D, D], sums the 72 projected rows
    else:
        raise ValueError(f"cross-term form must be 'scrambled' or 'clean', got {form!r} [DECISION D-36]")
    return F.softmax(a, dim=-1)  # [B_q, N+1, D, D]


def vip_module_forward(module: nn.Module, query: torch.Tensor, supports: torch.Tensor, prototype: torch.Tensor,
                       form: str) -> torch.Tensor:
    """`module(query, supports, prototype)` of an inherited PEM or PDM, with the cross-term in `form`.

    query F^q [B_q, P, D], supports F^s [N, K, P, D], prototype [B_q, N+1, D] -> [B_q, N+1, D]. Every line
    follows PEM (Eq.9-14) [VIPSEG models/vipseg.py:235-310] or PDM (Eq.15-18) [VIPSEG models/vipseg.py:338-405]
    and uses the module's own submodules; only `cross_attention` depends on `form` [DECISION D-36].
    """
    name = type(module).__name__
    if name not in MODULE_NAMES:
        raise TypeError(f"expected one of {MODULE_NAMES}, got {name}")
    pem = name == MODULE_NAMES[0]
    nway, kshot, pn, dim = supports.shape
    way = nway + 1  # background slot first
    residual = prototype  # [B_q, N+1, D]
    sup_all = torch.cat([supports.mean(0).unsqueeze(0), supports], dim=0)  # [N+1, K, P, D], slot 0 = mean of ways
    q = module.maxpool(query.transpose(1, 2)).transpose(1, 2)  # [B_q, 64, D] (Eq.9)
    s = module.maxpool(sup_all.flatten(0, 1).transpose(1, 2)).transpose(1, 2)  # [(N+1)*K, 64, D]
    s = s.unflatten(0, (way, kshot))  # [N+1, K, 64, D]
    proto = 0
    for i in range(kshot):
        que = module.map(q)  # Q' [B_q, 72, D]
        sup = module.map(s[:, i])  # S' [N+1, 72, D]
        new_proto = module.proto_map(prototype)  # [B_q, N+1, D]
        que_g, sup_g = que.transpose(1, 2) @ que, sup.transpose(1, 2) @ sup  # [B_q, D, D], [N+1, D, D]
        if pem:  # Eq.10-11 [VIPSEG models/vipseg.py:266-277]
            selfcor_q = module.reweight(que_g.unsqueeze(1)).squeeze(-1) / SCALE  # [B_q, 1, D]
            selfcor_s = module.reweight_s(sup_g.unsqueeze(0))[0, :, :, 0] / SCALE  # [N+1, D]
            proto_self_q = torch.sigmoid(selfcor_q) * new_proto  # [B_q, N+1, D]
            proto_self_s = torch.sigmoid(selfcor_s) * new_proto  # [B_q, N+1, D]
            proto_self = module.fc_qs(proto_self_s) + module.fc_qs(proto_self_q)  # [B_q, N+1, D]
            proto_self = module.layer_norm_qs(proto_self)  # [B_q, N+1, D]
        else:  # Eq.15-16 [VIPSEG models/vipseg.py:369-380]
            delta_g = que_g.unsqueeze(1) - sup_g.unsqueeze(0)  # [B_q, N+1, D, D]
            selfcor = module.reweight(delta_g)[..., 0] / SCALE  # [B_q, N+1, D]
            proto_self = torch.sigmoid(selfcor) * new_proto  # [B_q, N+1, D]
        attn = cross_attention(que, sup, form)  # [B_q, N+1, D, D] (Eq.12 / Eq.17)
        proto_cross = torch.matmul(attn, new_proto.unsqueeze(-1)).squeeze(-1)  # [B_q, N+1, D] (Eq.13)
        output = module.fc(proto_cross + proto_self)  # [B_q, N+1, D] (Eq.14 / Eq.18)
        output = module.layer_norm(output + residual)  # [B_q, N+1, D]
        proto = proto + output / kshot  # [B_q, N+1, D]
    return proto


class CrossFormModule(nn.Module):
    """An inherited PEM/PDM whose cross-term form can be switched between passes [DECISION D-36].

    Wraps the entries of a loaded VIP-Seg model's `vip_module` for C4; the wrapped module and its class are
    not edited, and `native` calls it unchanged.
    """

    def __init__(self, module: nn.Module, form: str = "native"):
        super().__init__()
        self.inner = module
        self.form = form

    def forward(self, query: torch.Tensor, supports: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        if self.form == "native":
            return self.inner(query, supports, prototype)  # [B_q, N+1, D]
        return vip_module_forward(self.inner, query, supports, prototype, self.form)  # [B_q, N+1, D]


class VIPStage(nn.Module):
    """One VIP-Seg reasoning step with the signature of `EPPMStage`.

    `module(query [B_q, 2048, D], supports [N, K, 2048, D], prototype [B_q, N+1, D]) -> [B_q, N+1, D]`
    is VIP-Seg's own call [VIPSEG models/vipseg.py:155-158]; odd steps add the outer residual VIP-Seg
    applies to the PDM output [VIPSEG models/vipseg.py:157]. `module` is injected in tests, where
    `models.vipseg` cannot be imported. `cross_form` [DECISION D-36]: `native` calls the module,
    `scrambled` / `clean` run `vip_module_forward` on its weights; the parameters are the same in all three.
    """

    def __init__(self, step: int = 0, module: Optional[nn.Module] = None, cross_form: str = "native"):
        super().__init__()
        if cross_form not in CROSS_FORMS:
            raise ValueError(f"cross_form must be one of {CROSS_FORMS}, got {cross_form!r} [DECISION D-36]")
        self.step = step
        self.outer_residual = step % 2 == 1
        self.cross_form = cross_form
        self.module = build_vip_module(step) if module is None else module

    def forward(self, p_prev: torch.Tensor, f_s: torch.Tensor, f_q: torch.Tensor) -> torch.Tensor:
        """p_prev [B_q, N+1, D], f_s [N, K, 2048, D], f_q [B_q, 2048, D] -> P^t [B_q, N+1, D]."""
        if p_prev.dim() != 3 or p_prev.shape[0] != f_q.shape[0] or p_prev.shape[1] != f_s.shape[0] + 1:
            raise ValueError(f"P^(t-1) {tuple(p_prev.shape)} does not match F^s {tuple(f_s.shape)} and "
                             f"F^q {tuple(f_q.shape)}")
        if self.cross_form == "native":
            out = self.module(f_q, f_s, p_prev)  # [B_q, N+1, D]
        else:
            out = vip_module_forward(self.module, f_q, f_s, p_prev, self.cross_form)  # [B_q, N+1, D]
        return out + p_prev if self.outer_residual else out  # [VIPSEG models/vipseg.py:157]
