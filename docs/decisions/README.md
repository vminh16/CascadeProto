# Decision log (L3)

One file per decision, `D-nn_<slug>.md`. Each file keeps the decision exactly as it was recorded:
- the heading with its status;
- the problem and its evidence;
- what the decision does;
- the rules fixed before any run;
- amendments, which are dated and never rewritten;
- the outcome.

[`docs/spec/00_SOURCES_AND_DECISIONS.md`](../spec/00_SOURCES_AND_DECISIONS.md) §2 sets where these decisions rank
among the sources: paper (L1), then pinned VIP-Seg code (L2), then these decisions (L3). Its §4 is the index, one row
per decision with a preliminary result.

The files were split verbatim out of §4 of 00 on 2026-09-28. Every line of the old section is in exactly one file.

## Adding or changing a decision

1. **Before any code,** write `D-nn_<slug>.md` with the next free number. Start it with the heading line
   `### D-nn — <title> · \`<status>\``, and include:
   - the problem, with evidence;
   - what it does;
   - its rules;
   - what it affects.
2. Add one row to the index in 00 §4. Add one line to the change log in 00 §7.
3. After the run, append the outcome to the decision's file. Replace that row's "Preliminary result" in 00 §4 with
   one line of the outcome, copying its numbers exactly.
4. When the rules change before the run, add a dated **Amendment**. Do not edit the earlier text.

Code and specs cite a decision as `[DECISION D-nn]`, without a path.
