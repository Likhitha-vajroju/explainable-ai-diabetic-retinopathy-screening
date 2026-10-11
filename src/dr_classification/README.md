# Experimental diabetic retinopathy classification

This package implements five-class transfer-learning experiments using
torchvision (`resnet18` by default; `resnet50` and `efficientnet_b0` are
configurable). The model replaces the pretrained classification head with five
outputs and fine-tunes the network. The default ResNet-18 is a relatively
resource-conscious starting point; larger architectures require more compute.
With `pretrained: true`, torchvision downloads ImageNet weights on first use if
they are not already cached. For offline smoke tests, set `pretrained` to
`false`; random initialization is not a meaningful evaluation.

The package reads the project's existing `split_metadata.csv`, uses
`FundusClassificationDataset` to preprocess images with
`src.preprocessing.image_pipeline.preprocess_image`, and respects the
group-separated `train`, `validation`, and published `test` partitions. Training
fits on `train`, chooses the checkpoint by validation macro F1, and does not
iterate test images. Evaluation defaults to the held-out test split. Split CSV
rows and dataset files are read-only.

## Labels and responsible interpretation

The default configuration maps string codes `"0"` through `"4"` to the matching
indices and names them No DR, Mild, Moderate, Severe, and Proliferative DR.
The source dataset README notes that its raw code meanings have not been
independently verified against authoritative documentation. Confirm that the
selected dataset and its label definitions are appropriate before interpreting
model outputs as those severity levels. The default mapping is an explicit
experiment setting, not evidence that labels were clinically validated.

All reported metrics are measured on the provided data and are experimental.
They do not establish clinical validity or performance in other populations,
devices, or workflows. Confidence is the maximum softmax probability and is
not calibrated. No sensitivity, specificity, or accuracy target is asserted.
This is not a diagnostic service, and its outputs must not direct individual
care.

## Prepare the split metadata

Place the dataset as described in [`data/README.md`](../../data/README.md), then
audit it and create grouped split metadata from the repository root:

```powershell
python -m src.preprocessing.dataset_cli `
  --dataset-root "data/aptos-20152019-preprocessed" `
  --output-dir "data/metadata" `
  --seed 42 `
  --validation-fraction 0.15
```

The generated `split_metadata.csv` assigns published training-source samples
to train/validation and preserves the published test source as test. Inspect
the audit and grouping warnings before training. In particular, opaque
identifiers use sample-level groups because patient IDs cannot be inferred for
them, so patient leakage cannot be ruled out.

## Train

Install declared dependencies in the selected Python environment:

```powershell
python -m pip install -r requirements.txt
```

Run training (ImageNet pretrained weights require a network connection on
first use):

```powershell
python -m src.dr_classification.train `
  --config "src/dr_classification/config.json" `
  --dataset-root "data/aptos-20152019-preprocessed" `
  --split-csv "data/metadata/split_metadata.csv" `
  --output-dir "models/checkpoints/dr_resnet18"
```

Edit `config.json` to set architecture, epochs, batch size, learning rate,
seed, preprocessing parameters, worker count, or device (`auto`, `cpu`, or
`cuda`). To use a different architecture, also choose a matching checkpoint
output directory. The training split must include all five classes. Class
imbalance is addressed with inverse-frequency class weights in the
cross-entropy loss, calculated only from training labels; validation/test
labels do not influence those weights. Python, NumPy, PyTorch, and the shuffled
training data loader are seeded. Input pixels use the existing preprocessing
pipeline's `[0, 1]` RGB output followed by standard ImageNet channel
normalization for the pretrained torchvision backbone.

The default output directory (ignored by Git) contains:

- `best_model.pt`: model weights and configuration selected by validation macro
  F1; no test metrics are used for selection.
- `history.csv`: per-epoch train and validation loss, accuracy, macro
  precision, macro recall, and macro F1.
- `training_curves.png`: those measured train/validation metrics by epoch.

## Evaluate

Evaluate the selected checkpoint once on the held-out test partition:

```powershell
python -m src.dr_classification.evaluate `
  --checkpoint "models/checkpoints/dr_resnet18/best_model.pt" `
  --dataset-root "data/aptos-20152019-preprocessed" `
  --split-csv "data/metadata/split_metadata.csv" `
  --output-dir "reports/dr_classification"
```

Use `--split validation` only for development, not final reporting. The command
calculates loss, accuracy, macro precision, sensitivity/recall, one-vs-rest
specificity, macro F1, per-class support and metrics, five-class one-vs-rest
ROC-AUC, and a five-class confusion matrix. ROC-AUC is reported as undefined
for a class when the evaluated split does not contain both positive and
negative examples; the macro AUC averages only classes for which AUC is
defined.

Binary referable DR is defined here as levels 2/3/4, with levels 0/1
non-referable. Sensitivity, specificity, precision, F1, and the binary
confusion matrix group the **unchanged five-class argmax**; no model
predictions or thresholds are altered. Binary ROC-AUC uses the sum of
probabilities assigned to levels 2/3/4 as its continuous score. The requested
>90% sensitivity and >85% specificity are target requirements only: evaluation
compares actual measured values with those strict cutoffs and reports honestly
whether each is met.

Outputs include metrics JSON with ROC coordinates and train/validation/test
class counts, a labeled confusion-matrix CSV, a four-panel PNG dashboard
(row-normalized multiclass confusion matrix, binary confusion matrix, five
one-vs-rest ROC curves, and referable ROC curve), and a Markdown report with
setup, class imbalance, measured results, target status, and limitations.
No metrics are prefilled or fabricated.

## Evaluation interpretation and limitations

The dataset documentation reports raw label-code definitions that have not
been independently verified against authoritative clinical documentation,
mixed identifiers with sample-level fallbacks, and an unknown dataset license.
These affect interpretation, reuse, and potential leakage. Other possible
bias sources include camera/site and acquisition differences, image quality,
compression, disease-spectrum and demographic representation, selection bias,
class imbalance, and label noise. Some subgroup metadata may not be available
to measure these effects. Review the dataset audit and report's per-split class
counts.

A held-out benchmark measures only performance on that dataset split under its
label and sampling assumptions. It is not clinical validation. Clinical
validation additionally requires appropriately sourced and independently
reviewed labels, representative external cohorts, pre-specified protocols,
calibration and subgroup analysis, workflow evaluation, and clinical
governance. Neither satisfying the >90% sensitivity / >85% specificity targets
on this benchmark nor a high ROC-AUC alone establishes clinical safety or
generalization.

## Infer on one image

```powershell
python -m src.dr_classification.inference `
  "C:\path\to\fundus.jpg" `
  --checkpoint "models/checkpoints/dr_resnet18/best_model.pt"
```

The JSON result contains `predicted_dr_level`, `predicted_label`, all five
`class_probabilities`, and `raw_confidence` (the largest softmax probability).
An optional `--output "prediction.json"` saves a new JSON file and refuses to
overwrite an existing file.

## Tests

From the repository root:

```powershell
python -m unittest discover -s tests -p "test_dr_classification.py" -v
```

The inference-format tests check five probabilities, their sum, the predicted
class, and the uncalibrated-confidence description. The suite does not claim
model accuracy.
