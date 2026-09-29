# Phase 16 D-49 and D-48 arm A: results (2026-09-29)

Trained on the RTX 3090 instance, three runs together on CR's schedule (batch 1, 24,000 updates, 13 validations):
CB = CR + `--condition_balance 0.5` (D-49), A0 and A1 = the correlation head, `stage_type=corr`, 2 layers, seeds 0 and
1 (D-48 amendment 6). Wall time: CB 2 h 52 min, A0 and A1 3 h 43 min each (sharing the GPU). All `last.pt` read by
`experiments/d49_eval.py` on 7 draws: valid, fixed100, random600 × 3, leak-free, valid_raw; LP re-selected on each
checkpoint's valid (every checkpoint chose `lp_feat_k16_b0.99` for both seeds). CR's rows reproduce the earlier
measurements exactly (fixed100 54.84 / 55.74 / 57.63 / 58.55; random600 U+both+LP 58.37 / 60.36 / 59.99; leak-free
U+both+LP 33.15). Files: `<name>_<draw>.json`, `_counts.npz`, `_cond.npz`, `lp_<name>.json`, `train_<name>_S1.log`,
`d49_full.log`, `decideA_rerun.log`, `test_<name>.log`. 177 files copied, md5 checked.

## Validation during training (mIoU, `last` / best)

| run | last | best | curve |
| :--- | ---: | ---: | :--- |
| CR (D-37) | 55.17 | 55.82 (ep 48) | 48.6 → 55.8, rising |
| CB | 48.34 | 52.34 (ep 20) | 52.3 at ep 16–20, then 46–50 |
| A0 | 54.27 | 54.27 (ep 50) | 49.8 → 54.3 |
| A1 | 52.99 | 53.98 (ep 40) | 38.2 at ep 12, then 50–54 |

## Test (mIoU, `last.pt`)

| draw | row | CR | CB | A0 | A1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| fixed100 | model | 54.84 | 49.19 | 54.47 | 53.10 |
| | U | 55.74 | 49.93 | 55.04 | 53.34 |
| | U + both | 57.63 | 50.62 | 56.12 | 54.09 |
| | model + LP | 55.72 | 50.13 | 55.26 | 53.57 |
| | U + both + LP | 58.55 | 51.87 | 57.91 | 55.73 |
| random600 0 / 1 / 2 | U | 55.55 / 57.85 / 57.18 | 50.01 / 50.89 / 49.44 | 55.04 / 55.79 / 55.09 | 53.71 / 55.21 / 53.68 |
| | U + both + LP | 58.37 / 60.36 / 59.99 | 51.51 / 52.77 / 50.86 | 57.68 / 58.43 / 58.96 | 55.84 / 57.05 / 56.24 |
| leak-free | model | 28.34 | 25.17 | 29.28 | 28.49 |
| | U | 33.11 | **35.22** | 31.98 | 31.48 |
| | model + LP | 28.62 | 25.11 | 29.47 | 28.61 |
| | U + both + LP | 33.15 | **35.36** | 32.75 | 32.24 |

Condition recall on fixed100 (mean over the six test classes):

| row | CR own / other | CB own / other |
| :--- | :--- | :--- |
| model | 0.770 / 0.096 | 0.705 / 0.239 |
| U | 0.827 / 0.177 | 0.770 / **0.458** |

Leak-free, CB − CR, paired bootstrap: model −3.18 [−4.04, −2.29]; U **+2.11 [+1.43, +2.80]**; U + both +1.95
[+1.10, +2.82]; U + both + LP +2.21 [+1.23, +3.19].

## Rules

**D-49** (`d49_full.log`):

| rule | outcome |
| :--- | :--- |
| D49.1 mechanism | **fails**: median own-class abundance α of other points −0.018 → −0.021 (bar 0.10); the recall half holds (0.177 → 0.458, bar +0.05) |
| D49.2 leak-free | **holds**: U +2.11 [+1.43, +2.80] |
| D49.3 standard | fails: fixed100 −5.81 [−6.90, −4.74]; random600 −5.54 / −6.96 / −7.74 |
| verdict | **D49.5 stop** (no mechanism by the registered measure) |

**D-48 arm A** (`decideA_rerun.log`; the full run's own call lacked A1, see below):

| arm | A model+LP − CR U+both+LP |
| :--- | :--- |
| A0 | fixed100 −3.28 [−4.48, −2.12]; random600 −3.94 / −5.35 / −5.49; leak-free −3.67; other recall 0.084 → 0.115 |
| A1 | fixed100 −4.98 [−6.17, −3.81]; random600 −6.05 / −5.38 / −6.05; leak-free −4.53; other recall 0.084 → 0.090 |
| verdict | **neck not kept** (seed spread 1.70, smallest effect −4.98) |

## Reading

* **Condition balance moves the sparse points, which nothing on CR's features could.** The mean rule's recall on
  other-condition points rises from 0.18 to 0.46, and leak-free rises by 2.1 points with a CI clear of zero: the first
  leak-free gain on the clean base since D-37. The lever is the training distribution (queries always dense in their
  own class), not the head or an inference rule.
* **The price is the standard protocol**: −5.5 to −7.7 on fixed100 and random600, and −3.2 on leak-free for the
  model row. The head learned to call the target more often; the features (U row) gain where the density shortcut is
  removed and lose where it is present. q = 0.5 thins half the queries, far more than the test distribution does.
* **The abundance α is not a valid mechanism measure.** d49_eval also reports α for own-condition points: the median
  is −0.018 for CR and −0.021 for CB, the same as for other points, although the mean rule labels 83 % of own points
  correctly. For an own point the context mean c(x) is made of its own class, so c(x) and the class mean are nearly
  collinear and the 2 × 2 fit gives the class's share to γ. α did not move while other recall rose 2.5-fold. D49.1's
  α half therefore cannot see the mechanism it was registered to see; the rule is applied as registered (D49.5), and
  the measure is flagged here.
* **This corrects P11.6's reading** (`results/phase16_d48/SUMMARY.md`): "sparse points carry no own-class signal,
  the information is lost in the representation" rested on α ≈ 0 for other points, without α for own points as a
  control. With the control, α ≈ 0 says nothing, and CB shows the sparse-point signal can be learned.
* **Arm A**: the correlation head is slightly better than PEM/PDM as a head on leak-free (model+LP 29.47 / 28.61 vs
  28.62) but loses 3–6 points against CR's full inference stack everywhere, with a 1.7-point seed spread. Closed.

## Incidents

* `run_d49.sh` looked for A1 under `…_corr_b1_qrandom_seed1`; `train.py` writes `…_noadrm_seed1_corr_…`. The full run
  trained A1 but skipped its evaluation; it was run by hand (`mu_a1.log`, `test_a1.log`) and `decideA` rerun with
  both seeds. Fixed in commit 22a10e3.
* Scoring was slow (single-threaded CPU, about 3 h per checkpoint for the LP selection on 1,500 valid episodes).
  Leak-free and valid_raw for CR and CB were also run in separate processes to read D49.1–D49.2 early; their numbers
  equal the main processes' exactly.
