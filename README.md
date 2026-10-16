# Explainable AI for Diabetic Retinopathy Screening in Rural India

A research-oriented project to explore how AI can support diabetic retinopathy (DR) screening and referral workflows in settings with limited access to ophthalmologists. It does not contain a trained model or clinical results and is not a clinically validated screening application.

## The real-world problem

Diabetic retinopathy can threaten sight when it is not identified and referred for care in time. Screening programs face practical constraints, especially in rural and underserved communities:

- Fundus photographs may be blurred, poorly illuminated, or otherwise unsuitable for assessment.
- Specialist ophthalmologists are not consistently available where screening takes place.
- A prediction without understandable evidence can be difficult for clinicians to assess and trust.
- Screening tools must fit telemedicine workflows and be designed for reliable operation at scale.

## Proposed solution

The intended system is an AI-assisted screening and referral-support platform. A future workflow may assess whether an image is suitable, analyze retinal findings, estimate DR severity, and present interpretable evidence with a review-oriented report. It is intended to support—not replace—qualified clinical judgment. Any future components will need appropriate technical, clinical, privacy, and deployment evaluation before use with patients.

## Major system modules

| Module | Intended responsibility |
| --- | --- |
| Preprocessing | Prepare fundus images consistently while retaining traceability to the input. |
| Image quality assessment | Identify images that may be unsuitable for automated analysis and support a recapture or human-review pathway. |
| DR classification | Experimental five-level transfer-learning classifier; outputs are not clinically validated. |
| Explainable AI | Present model evidence in a way that can be inspected, with limitations clearly stated. |
| Lesion and retinal analysis | Explore analysis of clinically relevant retinal structures and lesions. |
| Reporting | Combine image, classification, confidence, and visual-evidence outputs into review-oriented JSON, Markdown, and PDF reports without presenting them as a diagnosis. |
| Screening-result history | Store minimal, optional screening history in a local SQLite database for lookup by anonymous patient ID and display previous screening results without implying confirmed disease progression. |
| Utilities | Hold shared, non-domain-specific helpers as the project grows. |
| Evaluation | Define reproducible technical evaluation and, later, appropriate clinical evaluation protocols. |
| Frontend and simulation | Reserve space for future user workflows and deployment/workflow simulation. |

## Intended users

The intended users are screening-program staff, trained health workers, optometrists, ophthalmologists, telemedicine teams, and researchers evaluating screening and referral-support workflows. The project is not intended for self-diagnosis or unsupervised treatment decisions.

## Technology stack

The current project uses Python, Pillow, and NumPy for image handling and preprocessing, PyTorch/torchvision for experimental transfer learning and Grad-CAM visualizations, FastAPI and SQLite for the local API and optional screening-result history, and React/Vite for the review UI.

## Project architecture

```text
data/                         Dataset documentation; data is not stored in Git
src/
  preprocessing/               Image preparation
  image_quality_assessment/    Image suitability research
  dr_classification/           Experimental five-class transfer learning
  explainable_ai/              Grad-CAM visualizations for classifier attention
  lesion_retinal_analysis/     Standalone vessel/disc candidates and annotation overlays
  reporting/                   JSON, Markdown, and PDF screening reports
  utils/                       Shared utilities
models/                        Model artifact documentation; weights are not stored in Git
evaluation/                    Evaluation plans and future evaluation code
reports/                       Report documentation and future generated reports
frontend/                      React/Vite screening review interface
data/                           Local prototype data; screening history database is ignored by Git
simulation/                    Workflow and deployment simulation
tests/                         Automated tests
docs/                          Project and research documentation
```

The optional screening-history module stores only a pre-assigned anonymous ID,
screening timestamp, predicted DR level, referable status, model confidence,
image-quality score, and optional reviewer decision. Uploaded images and
unnecessary personal information are not saved in screening history. Records
are kept in `data/screening_history.sqlite3` by default (or at the path set by
`DR_HISTORY_DB`). Treat this local prototype database as sensitive and use only
anonymous identifiers. See [`frontend/README.md`](frontend/README.md) for the
history workflow and local setup.

After analysis, a qualified reviewer can record whether the AI prediction was
accepted or modified, or flag the image for manual review or as ungradable.
The anonymous case ID and screening/review timestamps are stored with that
decision. Reviewer controls do not authenticate or authorize users in this
prototype; an ophthalmologist or other qualified clinician remains responsible
for final clinical interpretation. AI predictions and visual evidence support
review but do not make autonomous clinical decisions.

## Responsible AI and medical disclaimer

This is a research and educational project, not a medical device or a clinically validated screening service. The image quality score is an engineering screening metric with configurable heuristic thresholds; neither its score nor its thresholds have been clinically validated. It does not diagnose diabetic retinopathy, determine treatment, or replace examination by a qualified healthcare professional. Do not use it to make decisions about an individual's care. Future work must address image-quality failures, uncertainty, explainability limits, dataset provenance and representation, privacy, security, local clinical workflows, and validation across intended populations and devices. All patient-facing use would require appropriate clinical oversight and regulatory and institutional review.

The DR classifier is an experimental research component. Its dataset metrics
and softmax confidence are not clinical validation, calibrated confidence, or
evidence of performance in other settings. The current dataset documentation
also requires verification of the label-code meanings before interpreting
outputs as clinical severity levels.

## Image quality gate

The image quality assessment module reports separate focus, illumination,
contrast, field-of-view, and basic clipping-artifact measurements. Its
`ACCEPTABLE`, `BORDERLINE`, or `UNGRADABLE` result is intended to prevent poor
images from being silently forwarded into future automated analysis. A caller
should route borderline or ungradable images for review or recapture rather
than treat the score as a DR prediction. The configurable default thresholds
are engineering starting points, not medical cutoffs, and require validation
before use in any clinical workflow. See
[`src/image_quality_assessment/README.md`](src/image_quality_assessment/README.md)
for the method, configuration, CLI, and manual test commands.

## Retinal image preprocessing

The preprocessing package includes a configurable, one-image pipeline for
aspect-preserving resizing, retinal-background handling, illumination
normalization, luminance-only CLAHE, optional noise reduction, and normalized
pixel output. Its demo saves a separate source-image copy, enhanced image, and
before/after comparison without overwriting existing files. These transformations
are engineering tools only; no improvement in diagnostic accuracy is claimed.
See [`src/preprocessing/README.md`](src/preprocessing/README.md) for rationale,
configuration, and manual test commands.

## Experimental DR classification

The project includes ResNet transfer-learning training, validation-based model
selection, held-out test evaluation, five-class and binary referable metrics,
validation-fitted temperature scaling, reliability diagrams, screening-result
formatting, ROC-AUC, evaluation reports/plots, training curves, and single-image
inference.
The >90% sensitivity and >85% specificity values are targets, not guaranteed
results; actual evaluation values are reported without modifying predictions.
No performance results are prefilled or claimed; metrics are produced only by
running training/evaluation on local data. Checkpoints and evaluation outputs
are excluded from Git. Referable levels 2–4 recommend ophthalmologist review
under this project definition; calibrated confidence is not clinical certainty.
See
[`src/dr_classification/README.md`](src/dr_classification/README.md) for dataset
setup, training, evaluation, inference, and limitations.

## Development roadmap

1. **Foundation:** establish repository structure and project documentation.
2. **Data and protocol planning:** document permitted datasets, provenance, privacy, labeling assumptions, and evaluation design before handling data.
3. **Image pipeline and quality assessment (in progress):** implement and test image handling, enhancement, and suitability checks.
4. **Research models (experimental classifier implemented):** evaluate classification and retinal-analysis components with reproducible experiments.
5. **Interpretability and reporting (Grad-CAM implemented):** investigate evidence presentation and human review workflows; Grad-CAM attention is not lesion evidence.
6. **Evaluation and integration:** assess technical performance, failure modes, generalization, and workflow needs before any deployment consideration.

Each stage should be scoped, reviewed, and validated before proceeding. The
current scope includes dataset-preparation utilities, configurable retinal
image preprocessing, an engineering image quality assessment module, an
experimental DR classifier with calibration and referable screening support,
standalone retinal candidate analysis, Grad-CAM visualization, automated
JSON, Markdown, and PDF reports, and a local FastAPI/React screening review
application. Grad-CAM shows model attention and is not proof of a clinical
lesion or a clinical diagnosis. See [frontend/README.md](frontend/README.md)
for local startup instructions.
