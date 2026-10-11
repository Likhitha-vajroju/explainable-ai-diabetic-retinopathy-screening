# Explainable AI for Diabetic Retinopathy Screening in Rural India

A research-oriented project to explore how AI can support diabetic retinopathy (DR) screening and referral workflows in settings with limited access to ophthalmologists. It does not contain a trained model or clinical results and is not a screening application.

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
| Reporting | Organize analysis outputs for review and referral support without presenting them as a diagnosis. |
| Utilities | Hold shared, non-domain-specific helpers as the project grows. |
| Evaluation | Define reproducible technical evaluation and, later, appropriate clinical evaluation protocols. |
| Frontend and simulation | Reserve space for future user workflows and deployment/workflow simulation. |

## Intended users

The intended users are screening-program staff, trained health workers, optometrists, ophthalmologists, telemedicine teams, and researchers evaluating screening and referral-support workflows. The project is not intended for self-diagnosis or unsupervised treatment decisions.

## Technology stack

The current project uses Python, Pillow, and NumPy for image handling and preprocessing, and PyTorch/torchvision for experimental transfer learning. Candidate technologies for later, separately scoped stages include model-interpretability tooling and an API/UI framework.

## Project architecture

```text
data/                         Dataset documentation; data is not stored in Git
src/
  preprocessing/               Image preparation
  image_quality_assessment/    Image suitability research
  dr_classification/           Experimental five-class transfer learning
  explainable_ai/              Future interpretability methods
  lesion_retinal_analysis/     Future retinal and lesion analysis
  reporting/                   Future review-oriented reporting
  utils/                       Shared utilities
models/                        Model artifact documentation; weights are not stored in Git
evaluation/                    Evaluation plans and future evaluation code
reports/                       Report documentation and future generated reports
frontend/                      Reserved for a future user interface
simulation/                    Workflow and deployment simulation
tests/                         Automated tests
docs/                          Project and research documentation
```

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
selection, held-out test evaluation, confusion-matrix/training-curve outputs,
and single-image inference. No performance results are prefilled or claimed;
metrics are produced only by running training/evaluation on local data.
Checkpoints and evaluation outputs are excluded from Git. See
[`src/dr_classification/README.md`](src/dr_classification/README.md) for dataset
setup, training, evaluation, inference, and limitations.

## Development roadmap

1. **Foundation:** establish repository structure and project documentation.
2. **Data and protocol planning:** document permitted datasets, provenance, privacy, labeling assumptions, and evaluation design before handling data.
3. **Image pipeline and quality assessment (in progress):** implement and test image handling, enhancement, and suitability checks.
4. **Research models (experimental classifier implemented):** evaluate classification and retinal-analysis components with reproducible experiments.
5. **Interpretability and reporting:** investigate evidence presentation and human review workflows.
6. **Evaluation and integration:** assess technical performance, failure modes, generalization, and workflow needs before any deployment consideration.

Each stage should be scoped, reviewed, and validated before proceeding. The
current scope includes dataset-preparation utilities, configurable retinal
image preprocessing, an engineering image quality assessment module, and an
experimental DR classifier. It does not include Grad-CAM or a frontend.
