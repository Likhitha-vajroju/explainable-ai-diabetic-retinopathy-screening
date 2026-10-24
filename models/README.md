# Model artifacts and requirements

This directory contains an experimental checkpoint at
`checkpoints/dr_resnet18/best_model.pt`, its training metrics in
`checkpoints/dr_resnet18/history.csv`, and a training-curve visualization in
`checkpoints/dr_resnet18/training_curves.png`. These artifacts are provided
for research and software demonstration, not clinical use. The metrics are
not clinical validation, and dataset provenance, label mapping, and training
configuration should be verified before drawing conclusions or redistributing
the artifacts.

The application can start without a checkpoint, but image analysis remains
unavailable until a compatible checkpoint is configured.

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
and label/data limitations. Do not present this checkpoint or its training
metrics as clinical validation.
