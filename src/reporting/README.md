# Automated screening report

`src.reporting.screening_report` combines the existing image-quality,
preprocessing, classifier, calibration, Grad-CAM, and retinal-analysis
components. It produces a machine-readable JSON result, a human-readable
Markdown report, and a PDF summary with available visual evidence.

```powershell
python -m src.reporting.demo `
  --dataset-root "data/aptos-20152019-preprocessed" `
  --checkpoint "models/checkpoints/dr_resnet18/best_model.pt" `
  --output-dir "reports/screening_demo"
```

The demo selects the first supported fundus image found under the local
dataset root. Checkpoint and dataset files are not included in the repository;
provide a trained checkpoint and locally available dataset. An optional
`--calibration-temperature` may be used only with a temperature fitted on the
validation split for the same checkpoint.

The JSON intentionally contains no patient-identifying or demographic fields
other than `patient_information: null`. It includes image dimensions,
engineering image-quality metrics, preprocessing status, class probabilities,
calibrated confidence when provided, project-defined referable status,
recommended next action, and artifact statuses/paths. Grad-CAM and retinal
analysis are optional: their errors are reported as unavailable and do not
discard the main classifier result. No lesion detector is claimed; the retinal
component emits vessel/disc candidates and the report notes that no lesion
annotations were supplied.

The report prominently states that this is an AI-assisted screening prototype,
not a definitive diagnosis or treatment decision. Ophthalmologist review is
recommended for referable cases. Model performance depends on dataset,
annotation, and validation conditions; the report does not claim clinical
validity or generalization.
