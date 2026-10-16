# Model artifacts and requirements

This directory is reserved for local model artifacts. Trained weights and
checkpoints are excluded from version control; no trained DR checkpoint is
distributed with this repository. The application can start without a
checkpoint, but image analysis remains unavailable until a valid one is
configured.

## Application checkpoint

The current classifier implementation supports `resnet18` by default, with
`resnet50` and `efficientnet_b0` as training configurations. The API expects a
checkpoint that can be loaded by `src.dr_classification.classifier` and whose
architecture, class mapping, image size, and preprocessing configuration
match the inference configuration. The default project configuration has five
outputs for Levels 0–4. Confirm dataset label meanings before interpreting
these as clinical severity categories.

Set the file path before starting the backend:

```powershell
$env:DR_CHECKPOINT = "models/checkpoints/dr_resnet18/best_model.pt"
```

For optional calibration, use only a finite positive temperature fitted on a
validation split for that same model checkpoint:

For example, if (and only if) the fitted validation temperature for the
configured checkpoint is `1.2`:

```powershell
$env:DR_CALIBRATION_TEMPERATURE = "1.2"
```

Do not copy that example unless it is the value actually fitted for the
checkpoint. When calibration is not configured, the API returns raw
confidence and reports calibrated confidence as unavailable. Calibration
does not alter the predicted grade or referral group.

See
[src/dr_classification/README.md](../src/dr_classification/README.md) for
dataset preparation, training, checkpoint selection, inference, evaluation,
and label/data limitations. Before distributing an artifact, document its
provenance, license, training configuration, dataset split, validation-fitted
calibration (if any), intended use, and limitations. Do not present a local
checkpoint or measured dataset metrics as clinical validation.
