### D-02 — Target of entropy gating · `LOCKED`

* **Problem.** Eq.10 gates "a feature vector x ∈ R^D", §1 speaks of suppressing "high-entropy background features", but the next sentence says "After entropy gating, we apply cross-attention between the query and support to further refine the prototype", and Eq.14/Eq.21 only reference `P^{t−1}`.
* **Decision.** Gate the incoming prototype channel-wise: `P^{t−1}_gated = P^{t−1} ⊙ g(P^{t−1})`, with one scalar θ per stage [PAPER §3.5 "each EPPM applies entropy gating independently"]. `P^{t−1}_gated` is the input to ψ in D-01. The residual in Eq.21 uses the **ungated** `P^{t−1}`, as printed.
* **Rationale.** Channel-wise gating composes naturally with the channel–channel attention of D-01, and it is the only reading consistent with Eq.14 and Eq.21 using `P^{t−1}`.
* **Known limitation.** With θ = 0.5 and natural log, `g ∈ [0.405, 0.731]`, so the gate is weak at initialisation (audit H4).
* **Outcome of the paper audit (2026-09-20, `docs/research/2026-09-20_paper_vs_code_audit.md`).** The
  rationale above does not hold. Both readings are consistent with Eq.14 and Eq.21 referencing
  `P^{t-1}`: if the gate applied to `F^q`/`F^s`, Eq.13's `φ(F^q)` would simply consume gated features
  and Eq.14 would be unchanged. Worse, under `gate_target = prototype` the printed Eq.14 multiplies
  `ψ(P^{t-1})`, the **ungated** prototype, and Eq.21's residual is also `P^{t-1}`, so `x_gated` — a
  symbol that appears in Eq.12 and **nowhere else in the paper** — would never be consumed and Eq.10–12
  would be dead. Our code only makes the gate matter by feeding ψ the gated prototype, which is a
  silent departure from Eq.14. The paper's own wording leans the other way: Eq.10 says "for a **feature
  vector** x ∈ R^D", Eq.12 says "the **gated feature** is", and §3.2 says each module "suppresses
  high-entropy background **features**".
* **Revised decision (2026-09-20).** `gate_target = features` is implemented and no longer raises: the
  gate is applied per point to `F^s` and `F^q` before Eq.13, and ψ receives the ungated `P^{t-1}`. The
  default stays `prototype` until a VM run separates them, because changing it changes every trained
  checkpoint's meaning. The diffusion branch of Eq.15–18 is ungated under both readings, since those
  equations name `F^q` and `F^s` with no mention of gating.
* **Ablation flag.** `gate_target = {prototype (default), features}`. Neither value raises.
