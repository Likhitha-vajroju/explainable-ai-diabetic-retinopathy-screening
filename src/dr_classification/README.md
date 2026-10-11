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
calculates loss, accuracy, macro precision, macro recall, macro F1, per-class
metrics/support, and a five-class confusion matrix. It saves metrics JSON,
confusion-matrix CSV, and a PNG visualization. It reports measured results
only; no metrics are prefilled or fabricated.

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
