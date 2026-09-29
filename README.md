# What a Moderation Budget Buys

Source for the main allocation method in *What a Moderation Budget Buys: Repair
Opportunities, Allocation Loss, and Charging*. Given development/test prediction
banks, this package fits RF and LR repair-gain estimators, runs exact full/prefix
allocation and ranked charging policies, and computes paired rerouted intervals.
It runs on CPU and does not need PyTorch.

This is a **source-only release**. It contains no datasets, split maps, labels,
prediction banks, checkpoints, sample fixtures, or saved results. Supply your
existing prediction banks through the connector below. All outputs must be
outside this source directory.

## Install and smoke test

Use Python 3.11 (the recorded environment used 3.11.14):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON=.venv/bin/python bash run.sh smoke
PYTHON=.venv/bin/python bash run.sh --help
```

`run.sh smoke` is CPU/offline. It constructs explicitly synthetic banks in a
system temporary directory and deletes them on completion. It checks exhaustive
small-instance allocation/frontiers, ties and zero budgets, guarded refunds,
duplicate instances, LR missing classes, shared draws, seed averaging, and all
three corpus connectors. It does not download data or reproduce paper results.
Do not run with Python's `-O` option; numerical checks must remain enabled.

## Prediction-bank connector

Provide one development and one test `.npz` file per frozen base-model seed.
Arrays must be numeric/boolean or NumPy strings; object arrays and pickle are
unsupported. Each pair must share a base-model checkpoint. Development data
fits the gain estimators; test references are used only after action admission
for refund charging and after allocation for evaluation.

`N` is the number of records. Action columns always mean `[none, depth1, depth2,
depth3]`, with nominal costs `[0,1,2,3]`.

| Array | Shape | Meaning |
|---|---|---|
| `ids` | `(N,)` strings | Unique opaque record IDs; no source text |
| `group` | `(N,)` strings | Exact **original-post-text** equality group; repeated groups are retained |
| `joint_success` | See below | Existing-reference complete correctness after each action, 0/1 |
| `t1_correct` | `(N,4)` | Existing-reference stage-one correctness, 0/1 |
| `depth` | `(N,)` | True applicability depth: TAMA 1 or 3; external 1, 2 or 3 |
| `p1`, `p2` | See below | Native pre-review probability vectors |

All action outcomes must use the paper's reference-correction convention:
correct distributions stay unchanged; an incorrect checked field is replaced
by its existing reference, and downstream predictions are recomputed. Full
review, and review through every applicable field, must complete the output.
Joint correctness must be nondecreasing with depth; checked stage one is always
correct. The connector validates these invariants, shapes and probability sums.
It cannot establish from numeric banks whether upstream fitting used proper
partitions or whether group IDs represent exact text; the input provider must
establish that provenance.

TAMA fields:

- `p1`: `(N,3)`, class order `non-abusive`, `targeted-abusive`,
  `unidentified-targets`.
- `p2`: `(N,12)`, order `death_threat`, `sexual_assault`, `sexual_explicit`,
  `physical_harm`, `radiation_of_threats`, `attacks_on_credibility`,
  `misogynistic`, `homophobic`, `religious`, `political_sectarian`, `racist`,
  `general`.
- `joint_success`: `(N,4,3)` with `thresholds=[.5,.8,1.]`, or `(N,4)` with a
  matching scalar `threshold`. The threshold is the span-F1 requirement for a
  complete output, not a token probability threshold.
- `gold_t1` of shape `(N,)` can replace `depth`: label 1 implies depth 3;
  labels 0/2 imply depth 1. If both are supplied they must agree.

OLID-BR/MOLD fields:

- `p1`: `(N,2)`, order `NOT, OFF`; `p2`: `(N,2)`, order `UNT, TIN`;
  `p3`: `(N,3)`, order `IND, GRP, OTH`.
- `characters`: `(N,)`, nonnegative integer counts of original-post characters.
- `joint_success`: `(N,4)`; the CLI threshold is 0.
- Original external banks may provide `stage_correct` of shape `(N,4,3)`
  instead of `t1_correct`; its first-stage column is used.

Native features are reconstructed: TAMA uses three T1 probabilities, their
natural-log entropy, and `max(p1)*max(p2)` when native T1 predicts TA (otherwise
`max(p1)`). External corpora use the seven native probabilities, their three
stage entropies, and `log1p(characters)`. Optional `features` must agree with this
specification (tolerance `1e-7`); validated supplied values are retained to
preserve frozen-fit floating-point parity.

Optional metadata `tri_classes`, `types`, `classes1/2/3`, `action_ids`, `costs`,
`split`, `seed`, and `dataset` is checked when present. Without class metadata,
the connector **assumes exactly the orders above**. Scalar `dataset` values are
`tama`, `olid_br`, `mold2`. Optional `model`, `checkpoint_sha256`, and
`adapter_sha256` must match between development and test. These hashes document
identity; this package does not load or verify checkpoint bytes. Other fields
in original banks, including token diagnostics, are ignored.

No development/test IDs or exact-text groups may overlap. Across seeds, test
IDs, group membership and applicability depths must match in the same order.
TAMA keeps the original bank order, including exact-allocator ties. External
corpora use a stable ID sort. Do not reorder TAMA banks when reproducing frozen
allocations. Keep discordant duplicate references as distinct records sharing
one exact-text group. Never use historical relationship-derived groups.

For the supplied study snapshot, preserve TAMA's existing 4,647/1,549/1,550
train/dev/test membership; OLID-BR's complete-label counts are 4,055/1,014/1,689;
MOLD's are 2,275/569/510 after the study's existing missing-label and exact-text
overlap exclusions. No partition maps are bundled. The connector does not
regenerate partitions, infer missing labels, or treat missing labels as exits.
No reply graphs, parent posts, conversation inputs, or new annotation are used.

## Run the main analysis

Point the following variables at your externally supplied banks and a **new**
output directory outside this repository. `--dev` and `--test` follow `--seeds`
order. Paths shown here are placeholders, not bundled files.

```bash
C_INPUT_DIR=/path/to/predictions
C_OUTPUT_DIR=/path/to/new-tama-analysis
PYTHON=.venv/bin/python bash run.sh --corpus tama --threshold .5 \
  --seeds 17 29 43 59 \
  --dev "$C_INPUT_DIR/seed_17/dev_scores.npz" "$C_INPUT_DIR/seed_29/dev_scores.npz" \
        "$C_INPUT_DIR/seed_43/dev_scores.npz" "$C_INPUT_DIR/seed_59/dev_scores.npz" \
  --test "$C_INPUT_DIR/seed_17/test_scores.npz" "$C_INPUT_DIR/seed_29/test_scores.npz" \
         "$C_INPUT_DIR/seed_43/test_scores.npz" "$C_INPUT_DIR/seed_59/test_scores.npz" \
  --out "$C_OUTPUT_DIR" --reps 2000 --workers 4
```

For OLID-BR or MOLD, use the corresponding banks and `--corpus olid_br` or
`--corpus mold`; omit `--threshold`. A single supplied fit can be evaluated with
`--seeds 17 --dev /path/dev.npz --test /path/test.npz`; this produces individual
seed estimates, not four-seed means. Lower `--reps` for a diagnostic run only.
TAMA thresholds `.8` and `1` run exact full/prefix arms only, matching the study.

The fixed protocol uses budgets `floor(.6*N)` and `floor(1.2*N)`. RF parameters
are 200 trees, maximum depth 5, minimum leaf size 15 and random state 17. LR fits
minimum repair depth with development-only standardization, L2 regularization,
`C=1`, `lbfgs`, tolerance `1e-4`, 2,000 maximum iterations, and no class weighting.
Already-correct probability is excluded from cumulative repair gain. A
convergence failure stops the run. Numerical library threads are limited to one
per worker; `--workers` controls at most eight concurrent seed jobs.

R0 ranks positive estimated gain per requested depth and keeps fixed charges.
R1 uses the same ranks and refunds inapplicable checks. R2 ranks by expected cost
with the same refund rules. Requested depth must fit the remaining budget
**before** applicability is revealed. Admitted instances are never upgraded.
TAMA expected cost is `1+(depth-1)*p_TA`; external cost is
`1+[depth>=2]*p_OFF+[depth>=3]*p_OFF*p_TIN`. Ties use ID then depth. Exact
allocation remains a separate comparator, and unchanged-action repricing is
reported as an accounting control.

Each of 2,000 shared draws samples exact-text groups with replacement, seed
20260923. Every duplicate copy is a separately charged instance. Allocation and
integer budgets are recomputed for the expanded batch. The same corpus draws
pair seeds, estimators, menus, budgets and charging arms. Four-seed contrasts
are averaged within a draw before 2.5/97.5 percentiles; original-batch points
and across-seed sample SD are separate. Intervals are pointwise and conditional
on frozen base models and development-fitted estimators, not retraining or
future-distribution guarantees. Fixed-action resampling is a separate control.

Outputs are `summary.csv` (absolute metrics and paired contrasts), one
`seed_<seed>.npz` (per-draw aggregates and original actions/charges) per seed,
`draws.npz`, `fits.json`, and `run.json` (settings, versions and input/source/output
hashes). Quantities labelled `per_record` are fractions, not percentage points;
multiply by 100 for paper-style percentages. Missing intervals for empty
strata remain missing. Fixed-charge frontier losses are intentionally undefined
for refunded policies. `main_protocol_settings` records only seed/draw settings;
it does not certify that supplied banks are the manuscript's original data.

## Scope and source provenance

The main numerical bodies are adapted from this project's frozen
`optimizer_probe.py`, `single_post_exact_group_audit.py`,
`single_post_pretrained_charging.py`, `c_budget.py`, `derive_audit.py`,
`c_rerouting.py`, and `c_rerouting_summary.py`. Local raw-text audits, historical
archive/source-hash gates, and workspace path discovery have been replaced by
the explicit connector. The allocator, estimator settings, row-order tie rules,
reservation/refund semantics, draw construction and paired aggregation are
retained. No third-party L2D code is used.

This package starts **after base-model prediction and reference-action export**.
It does not train TAMA/OLID-BR/MOLD classifiers, create those banks, rerun the
lexical gain-seed/cost sweeps, reproduce target-only or same-menu objective
appendix controls, or regenerate manuscript figures. Reproducing original
paper numbers requires the original externally supplied frozen prediction
banks and partitions. Synthetic smoke success is not evidence of reproduction.

Dataset attribution: TAMA is described by Liu et al., *TAMA: Target-Aware
Multilingual Abuse Detection by Cascaded Conditional Multi-Task Learning*
(ACL 2026). The OLID-BR dataset is available from
[dougtrajano/olid-br](https://huggingface.co/datasets/dougtrajano/olid-br), study
revision `84b0d7dd4309be677a47c535632a9398ff1897bd`; MOLD 2.0 from
[TharinduDR/MOLD](https://github.com/TharinduDR/MOLD), study commit
`6775a561c51987fbcd1c220109f492702b8cf0cd`. The audited external dataset releases
declare CC BY 4.0. Obtain authorized data and comply with its terms separately;
no dataset or upstream dataset source code is redistributed here.
