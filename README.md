# Results & Findings — Skin Cancer Detector

## Headline metrics

Binary classifier (malignant vs. benign), EfficientNet-B0 transfer learning,
trained 15 epochs on HAM10000 (best checkpoint: epoch 12, selected by a
composite of validation accuracy and sensitivity). Evaluated on a held-out,
stratified test set of 1,503 images.

| Metric | Value |
|---|---|
| Accuracy | 89.5% |
| Sensitivity (malignant recall) | 84.3% |
| Specificity (benign recall) | 89.3% |
| Malignant precision | 0.66 |
| ROC-AUC | 0.9516 |

Sensitivity was prioritized over raw accuracy throughout — with malignant
cases at ~20% of the dataset, a model that always predicted "benign" would
score ~80% accuracy while catching zero real cancer cases. The training loss
weighted the minority class accordingly, and checkpoint selection used a
composite score rather than accuracy alone, for the same reason.

The precision/recall balance (66% precision, 84% sensitivity) reflects a
model that errs toward flagging borderline cases rather than staying quiet —
the appropriate failure mode for a screening context, where a false alarm
costs a follow-up check and a missed cancer costs far more.

## Interpretability: a dataset-driven shortcut, found and quantified

Grad-CAM visualizations across a broad sample of test images revealed a
consistent pattern: for many correctly-classified benign images, the model's
attention landed **away from the visible lesion entirely** — on image
corners, edges, or background skin — rather than on the lesion itself.
Roughly a quarter of images showed a clean, lesion-localized heat pattern
resembling border-tracing (consistent with border irregularity, a real
ABCDE clinical warning sign), but a substantial fraction showed heat
concentrated somewhere structurally unrelated to the pathology.

Cross-referencing against the `dx_type` metadata field (how each diagnosis
was confirmed — biopsy/`histo`, clinical monitoring/`follow_up`,
multi-expert `consensus`, or `confocal` microscopy) surfaced a clear
candidate explanation: the off-lesion heat pattern appeared disproportionately
in `follow_up`-sourced images.

**Quantitative test:** restricting to only `nv` (benign nevus) images — so
the true label is held constant across the comparison — and comparing the
model's predicted malignant probability by `dx_type`:

| dx_type | n | Mean malignant prob. | False positive rate |
|---|---|---|---|
| follow_up | 565 | 0.0085 | **0.0%** |
| consensus | 67 | 0.140 | 10.4% |
| histo | 395 | 0.195 | **17.2%** |

Mann-Whitney U test (histo vs. follow_up): **p < 0.000001**.

Despite every image in this table having the identical true label (benign),
the model's output differs sharply and statistically significantly by how
the diagnosis was confirmed — a variable that has nothing to do with the
lesion's actual appearance from a medical perspective, but correlates with
it structurally through how the dataset was assembled.

**Why this happens:** `follow_up` is used clinically when a lesion is
already judged low-risk enough to simply monitor rather than biopsy, while
`histo` covers both real cancers and benign-but-ambiguous-looking lesions
that a clinician chose to excise rather than risk misjudging. That means
`histo`-confirmed benign cases are disproportionately the visually tricky
ones, while `follow_up`-confirmed cases are disproportionately the
unambiguous ones — a real clinical selection effect baked into the dataset,
likely compounded by systematic differences in how monitoring-visit photos
versus procedural-workup photos were captured (equipment, framing, lighting).
The model appears to have partly learned this proxy instead of relying
purely on lesion morphology.

## Why this matters

This is a materially different finding than "the model is biased" or "the
model is broken" — the aggregate test-set metrics above are real and hold up
under standard evaluation. The finding is narrower and more specific: **the
model's benign predictions on `follow_up`-sourced images are likely less
attributable to genuine lesion assessment than its predictions on
`histo`-sourced images**, even where both are correct. In a deployment
context without HAM10000's specific data-collection conventions, this
shortcut would not transfer — a genuinely malignant lesion captured in a
"monitoring-photo style" would not carry the same reassuring signal the
model has learned to associate with that image style here.

## Limitations

- Single public benchmark dataset (HAM10000); no external validation set
- No clinical review of predictions or Grad-CAM interpretations by a
  dermatologist
- `dx_type` confound identified here is specific to this dataset's
  collection protocol and may or may not generalize to other skin lesion
  datasets
- Small `consensus`/`confocal` subgroups (67 and ~100 images) limit
  confidence in per-group comparisons for those categories specifically
- This is a portfolio project evaluated on a public benchmark, not a
  validated diagnostic tool, and should not be presented or used as one