# Explainable AI for Diabetic Retinopathy Screening in Rural India

A research-oriented project to explore how AI can support diabetic retinopathy (DR) screening and referral workflows in settings with limited access to ophthalmologists. It does not contain a trained model or clinical results and is not a clinically validated screening application.

## The real-world problem

Diabetic retinopathy can threaten sight when it is not identified and referred for care in time. Screening programs face practical constraints, especially in rural and underserved communities:

- Fundus photographs may be blurred, poorly illuminated, or otherwise unsuitable for assessment.
- Specialist ophthalmologists are not consistently available where screening takes place.
- A prediction without understandable evidence can be difficult for clinicians to assess and trust.
- Screening tools must fit telemedicine workflows and be designed for reliable operation at scale.

## Prototype scope

This repository implements a local AI-assisted screening-result review
prototype: image-quality assessment, preprocessing, an experimental DR
classifier, optional visual evidence, screening reports, human reviewer
outcomes, and anonymous local screening history. It also includes separate
Python and Simulink resource-allocation models for exploring a hypothetical
telemedicine pathway. These components support research and demonstration;
they are not validated for clinical use and do not replace qualified clinical
judgment.

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
| Rural telemedicine resource simulation | Explore synthetic PHC-to-district screening workload, resource capacity, and waiting-time scenarios; not a real-world service-capacity claim. |
| Human review | Let reviewers record screening-result disposition; final clinical interpretation remains the responsibility of a qualified clinician. |
| Utilities | Hold shared, non-domain-specific helpers as the project grows. |
| Evaluation | Provide experimental dataset evaluation tools and documentation; no clinical evaluation is claimed. |
| Frontend | Provide a React/Vite local screening, review, and history interface. |
| Telemedicine simulation | Explore synthetic resource configurations in a separate Python discrete-event simulator and MATLAB/Simulink model. |

## Intended users

The intended users are screening-program staff, trained health workers, optometrists, ophthalmologists, telemedicine teams, and researchers evaluating screening and referral-support workflows. The project is not intended for self-diagnosis or unsupervised treatment decisions.

## Technology stack

The current project uses Python, Pillow, and NumPy for image handling and preprocessing, PyTorch/torchvision for experimental transfer learning and Grad-CAM visualizations, FastAPI and SQLite for the local API and optional screening-result history, React/Vite for the review UI, and Python/MATLAB scripts for resource-allocation simulation.

## Integrated screening workflow

The current local prototype wires the implemented modules together through a
FastAPI service and React/Vite interface:

1. The user selects a JPEG, PNG, TIFF, or BMP fundus image. The API validates
   the declared/content type, non-empty content, image dimensions, byte size
   (15 MB maximum), and pixel count (25 megapixels maximum).
2. The backend independently assesses image quality and preprocesses the
   source image. The UI can display the original or the processed image. These
   steps use engineering heuristics and are not clinically validated.
3. The configured five-class model returns a Level 0–4 predicted grade,
   probabilities, raw maximum softmax confidence, and a project-defined
   referable grouping (Levels 0–1 non-referable; Levels 2–4 referable).
   Optional temperature scaling reports calibrated confidence only when a
   matching fitted temperature is configured.
4. Grad-CAM and algorithmic retinal evidence are generated when available.
   They are displayed separately, may be unavailable, and are not confirmed
   lesions or proof of a diagnosis.
5. A screening report is generated in PDF, Markdown, and JSON formats and
   returned with the analysis response. Original uploads and generated report
   artifacts are temporary and are not kept as screening-history records.
6. The reviewer enters a pre-assigned anonymous case ID and selects a review
   outcome. At that point, the predicted screening result and review decision
   are recorded with screening and review timestamps in local SQLite. History
   lookup and the trend chart show screening-result history only, not confirmed
   disease progression.
7. The separate rural telemedicine simulation models resource queues from
   PHC acquisition through district ophthalmologist review. Its input
   parameters can be informed by local prototype benchmarks, but the simulator
   does not consume live application traffic or claim real-world service
   capacity.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for components, data flow,
storage boundaries, and integration gaps.

## Installation

### Backend

Use a Python version for which compatible PyTorch and torchvision packages
are available. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository does not include a trained checkpoint. Either provide a
compatible checkpoint using `DR_CHECKPOINT` or follow the dataset, training,
and evaluation instructions in
[src/dr_classification/README.md](src/dr_classification/README.md). Without a
loadable checkpoint, the health endpoint reports that the model is unavailable
and image analysis is disabled.

### Frontend

Install a current Node.js/npm environment, then run from the frontend folder:

```powershell
Set-Location frontend
npm ci
```

## Run the application

Start the API from the repository root in one terminal. Point `DR_CHECKPOINT`
to a locally available checkpoint that matches the project's configured
architecture and preprocessing:

```powershell
$env:DR_CHECKPOINT = "models/checkpoints/dr_resnet18/best_model.pt"
# Optional: only use a finite, positive temperature fitted for this checkpoint.
# $env:DR_CALIBRATION_TEMPERATURE = "1.2"
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
Set-Location frontend
npm run dev
```

Open the local Vite URL printed by the command (normally
`http://localhost:5173`). The backend defaults to the local Vite CORS origins.
Set `DR_CORS_ORIGINS` to an explicit comma-separated allowlist if using another
frontend origin.

### Use the review workflow

1. Select an authorized test fundus image and run analysis. Invalid images and
   unavailable models produce explicit errors.
2. Inspect image quality, preprocessing, original/processed image, DR level,
   confidence, referable status, available Grad-CAM/retinal evidence, and the
   generated report files.
3. Enter an anonymous case ID and choose one reviewer outcome: AI prediction
   accepted, AI prediction modified, image requires manual review, or image
   ungradable. The selected outcome and predicted result are saved to the
   local history database. This prototype does not authenticate or authorize
   reviewers; a qualified clinician remains responsible for final clinical
   interpretation.
4. Open **Screening history**, look up that anonymous ID, and inspect the
   screening results and their dates. Trend differences are not confirmed
   disease progression.

### Demonstrate the rural workflow simulation

Run the Python discrete-event simulation with a documented scenario:

```powershell
.\.venv\Scripts\python.exe simulation\telemedicine_simulation.py `
  --config simulation\scenario_100k.json `
  --output "$env:TEMP\telemedicine-100k.json"
```

The 100,000 annual volume is a planning input only, not a claim of real
clinical service capacity. The optional Simulink model is built and run
separately in MATLAB/Simulink; see
[simulation/README.md](simulation/README.md) for assumptions and parameter
mapping.

## Hackathon Demo (3–5 minutes)

Prepare beforehand: install dependencies, start the API and frontend, verify
the API health indicator is ready, and choose one authorized test image. A
loadable local checkpoint is required for live image analysis. No checkpoint
is distributed in this repository; without one, do not present mocked or
simulated values as model output.

1. **0:00–0:30 — Frame the prototype.** State that it is a research screening
   workflow, not a diagnostic product. Point out that the final clinical
   interpretation remains with a qualified clinician.
2. **0:30–1:30 — Upload and inspect.** Upload the test image and run analysis.
   Show image-quality measurements/flags, the original and enhanced/
   preprocessed view, the Level 0–4 screening prediction, confidence and
   calibration availability, and referable status. Do not imply enhancement
   improves diagnostic accuracy.
3. **1:30–2:30 — Explain and report.** Show Grad-CAM and available retinal
   evidence, explaining that attention/candidate overlays are not confirmed
   lesions. Open or download the generated PDF, Markdown, or JSON report.
4. **2:30–3:30 — Human review and history.** Enter a clearly synthetic
   anonymous demo ID (for example `DEMO-001`), choose a reviewer outcome, then
   look up the ID on **Screening history**. Identify the displayed trend as
   screening-result history, not disease progression.
5. **3:30–5:00 — Deployment simulation.** Run or show a pre-generated output
   from `scenario_100k.json`, then compare the stress/expansion scenarios.
   Explain which resource assumptions changed and that the annual volume is
   a planning scenario, not demonstrated operational capacity. If MATLAB/
   Simulink is installed and the model has been generated, show its daily
   queue/output plots; otherwise show the tested Python simulator and clearly
   state that Simulink execution was not verified.

If no trained checkpoint is available at demo time, demonstrate the UI's
explicit unavailable-model state, the independently runnable resource
simulation, and the test suite. Do not substitute canned grades, synthetic
patients, or mocked inference for a live model result.

## Project documentation map

- [Architecture](docs/ARCHITECTURE.md): integrated components and data flow.
- [Installation and application use](frontend/README.md): backend/frontend
  setup, configuration, and review interface details.
- [Classifier/model documentation](src/dr_classification/README.md) and
  [model artifact policy](models/README.md): architecture, training,
  checkpoint, inference, and limitations.
- [Evaluation documentation](evaluation/README.md): evaluation process and
  interpretation; no results are asserted here.
- [Telemedicine simulation](simulation/README.md): assumptions, parameters,
  outputs, benchmarking, and MATLAB/Simulink instructions.
- [Testing and known limitations](TESTING.md): verification commands and
  results for the engineering-quality pass.

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

This is a research and educational project, not a medical device or a
clinically validated screening service. Do not use it to make decisions about
an individual's care. The model does not diagnose diabetic retinopathy,
determine treatment, or replace assessment by a qualified healthcare
professional. An ophthalmologist or other qualified clinician remains
responsible for the final clinical interpretation. The reviewer interface
records workflow outcomes but provides no authentication or authorization.

The image-quality score and thresholds are engineering heuristics, not
clinically validated gradability measures. Poor-quality flags recommend
recapture and/or qualified review; they do not establish a clinical image
quality determination. Grad-CAM is a model-attention visualization.
Retinal-analysis overlays are algorithmic candidates. Neither is confirmed
lesion evidence. Confidence, including optional temperature-scaled
confidence, is not clinical certainty.

History records contain only an anonymous ID, screening timestamp, predicted
level, referable status, confidence, image-quality score, and optional reviewer
decision/timestamp. The local SQLite database is prototype storage, not a
clinical record system. Do not store identifiable patient data. Future work
must address privacy, security, dataset provenance and representation,
independent clinical validation, workflow safety, and regulatory and
institutional review before any patient-facing use.

The DR classifier is an experimental research component. Its dataset metrics
and softmax confidence are not clinical validation, calibrated confidence, or
evidence of performance in other settings. The current dataset documentation
also requires verification of the label-code meanings before interpreting
outputs as clinical severity levels.

## Image quality assessment

The image quality assessment module reports separate focus, illumination,
contrast, field-of-view, and basic clipping-artifact measurements. Its
`ACCEPTABLE`, `BORDERLINE`, or `UNGRADABLE` result is reported alongside the
model result. The current application does not hard-stop classification based
on this heuristic status; an `UNGRADABLE` result recommends recapture and
qualified review, and the model output should not be relied upon for that
image. Thresholds are engineering starting points, not medical cutoffs, and
require validation before use in any clinical workflow. See
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

## Further validation needed

The repository implements dataset-preparation utilities, configurable retinal
image preprocessing, engineering image-quality assessment, an experimental
DR classifier with calibration and referable screening support, heuristic
retinal candidate analysis, Grad-CAM visualization, JSON/Markdown/PDF reports,
a local FastAPI/React review application, and a separate rural
resource-allocation simulation. Implementation and software tests do not
establish clinical performance. Before any deployment consideration, the
project would need dataset provenance and label verification, pre-specified
independent validation, technical and clinical failure analysis, privacy and
security controls, and appropriate clinical, regulatory, and institutional
review. Grad-CAM attention is not proof of a lesion or diagnosis. The
simulation is not proof of real-world service capacity. See
[frontend/README.md](frontend/README.md) for application setup and
[simulation/README.md](simulation/README.md) for simulation assumptions,
usage, outputs, and limitations.
