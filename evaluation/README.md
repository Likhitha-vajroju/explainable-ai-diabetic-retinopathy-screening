# Evaluation protocol and interpretation

This project includes scripts for experimental, dataset-specific classifier
evaluation. It does **not** contain a tracked trained checkpoint, dataset,
completed evaluation report, or clinical validation result. No model
performance metrics are asserted in this repository. Do not infer model
quality from demo UI outputs, unit tests, or simulation results.

## Reproducing a model evaluation

First prepare the permitted dataset and grouped split metadata as documented
in [data/README.md](../data/README.md) and
[src/dr_classification/README.md](../src/dr_classification/README.md). Train
and select a checkpoint on the training and validation partitions only. Run
evaluation against the held-out test partition:

```powershell
python -m src.dr_classification.evaluate `
  --checkpoint "models/checkpoints/dr_resnet18/best_model.pt" `
  --dataset-root "data/aptos-20152019-preprocessed" `
  --split-csv "data/metadata/split_metadata.csv" `
  --output-dir "reports/dr_classification"
```

The evaluation implementation reports five-class metrics and per-class
support, one-vs-rest ROC-AUC where defined, confusion matrices, binary
referable metrics, and confidence/calibration summaries. Test metrics must
not be used to select model/checkpoint settings. Temperature scaling is fitted
on validation logits. A validation-only run reports calibration measured on
that same validation split as in-sample/optimistic.

The report records metrics from the selected data and split; it does not
establish clinical validity, performance on other datasets, or real-world
screening outcomes. The project-defined referable grouping is Levels 2–4
versus Levels 0–1. It groups the unchanged five-class argmax; it is not a
clinical diagnosis. Read the classifier documentation for unverified label
mapping, possible patient-level leakage, and dataset limitations before
interpreting any locally generated output.

## System and workflow checks

Automated tests validate software behavior (image handling, API errors,
preprocessing, report structure, history persistence, and simulation
invariants). They are not model-performance tests. API integration tests that
mock analysis/report generation verify service wiring and error responses but
do not verify actual predictions. The UI build does not constitute end-to-end
browser testing.

`simulation/benchmark_prototype.py` separately measures local image-quality
assessment and preprocessing, and optionally preprocessing-plus-prediction
when given a checkpoint. Its timing results are not clinical or production
benchmarks. Use measured times only as scenario inputs with environment,
image, device, and run details recorded. The telemedicine simulator evaluates
resource assumptions and queues, not diagnostic accuracy or patient outcomes.

## Reporting requirements for new experiments

Before running an experiment, specify dataset/version and license, label
definitions, inclusion/exclusion criteria, patient/site grouping, train,
validation, and test split rules, model/checkpoint provenance, preprocessing,
metrics and uncertainty, and planned subgroup analyses. Report sample counts
and class balance for every split. Preserve original outputs and disclose
failures and limitations; do not modify predictions or evaluation results to
improve reported performance. External and clinically governed validation
would be required before any clinical use.
