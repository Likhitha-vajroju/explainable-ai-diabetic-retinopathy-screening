# System architecture and integration

## Scope

This repository contains a local research prototype for AI-assisted diabetic
retinopathy (DR) screening-result review, plus a separate rural telemedicine
resource-allocation simulation. It is not a clinical system. The browser
application does not run the classifier; the FastAPI backend orchestrates the
Python analysis modules.

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| React review interface | `frontend/src/App.jsx` | Image selection, original/processed views, result/evidence/report display, human review entry, and history navigation. |
| History and trend view | `frontend/src/ScreeningHistoryPage.jsx` | Lookup by anonymous ID, chronological results, reviewer outcomes, and a screening-result trend chart. |
| FastAPI | `src/api/main.py` | Health/checkpoint state, image upload validation, analysis orchestration, response artifact encoding, history routes, and reviewer-decision update. |
| SQLite history store | `src/api/screening_history.py` | Stores minimal screening-result and review audit fields in the configured local database. |
| Image quality | `src/image_quality_assessment/` | Engineering measurements and flags for focus, illumination, contrast, field of view, and artifacts. |
| Preprocessing | `src/preprocessing/image_pipeline.py` | Deterministic configurable image transformations used before model input and for the processed-image view. |
| Classifier and screening grouping | `src/dr_classification/` | Five-class prediction, probability/confidence, optional temperature scaling, and project-defined referral grouping. |
| Explainability | `src/explainable_ai/` | Grad-CAM attention visualization for model review. |
| Retinal-analysis evidence | `src/lesion_retinal_analysis/` | Heuristic vessel/optic-disc candidates and overlays; not validated lesion detection. |
| Report generation | `src/reporting/screening_report.py` | Combines quality, preprocessing, prediction, referral, evidence, and disclaimer into JSON, Markdown, and PDF. |
| Resource simulation | `simulation/telemedicine_simulation.py` | Stochastic discrete-event resource queues and assumed stage outcomes. |
| Simulink builder | `simulation/build_simulink_model.m` | Builds a deterministic, daily-step fluid-queue model from scenario parameters. |
| Prototype timing | `simulation/benchmark_prototype.py` | Measures local image-quality and preprocessing time, and prediction time when a checkpoint is supplied. |

## Request and review data flow

```text
Browser selects fundus image
  -> POST /api/analyze (multipart upload)
  -> API validates type, bytes, dimensions, and pixel count
  -> report pipeline:
       image-quality assessment
       preprocessing/enhanced image generation
       five-level classifier prediction and referral grouping
       Grad-CAM (optional result)
       retinal candidate analysis (optional result)
       PDF + Markdown + JSON screening reports
  -> response contains result payload and base64-encoded temporary artifacts
  -> browser displays original/processed image, result, evidence, and reports
  -> reviewer supplies anonymous ID and one of four review outcomes
  -> POST /api/screening-history, then PUT
     /api/screening-history/{record_id}/review
  -> local SQLite stores screening result and reviewer decision/timestamps
  -> history page looks up the anonymous ID and plots past screening grades
```

The grade is the model's unchanged five-class argmax. Project-defined
referability groups Levels 0–1 as non-referable and Levels 2–4 as referable.
Optional temperature scaling changes the reported calibrated confidence, not
the predicted class or referral group. A poor-quality status is included in
the result and recommendation; it is not a validated clinical gradability
decision.

Visual evidence is explicitly best effort. Grad-CAM and retinal-analysis
failures can be surfaced as unavailable while a successful classifier result
is still returned. The evidence does not confirm lesions or explain the
prediction causally.

The browser retains the uploaded image for the active page session. The API
uses request-temporary storage for the upload and generated files; it returns
artifacts in the response rather than persisting them in history. SQLite
history is separate, stores no image or direct identity fields, and is local
prototype storage. A screening record is created when the reviewer submits
their first decision, not automatically on successful analysis. If creating
the record succeeds but the review update fails, the result record can remain
without a reviewer decision; the interface supports retrying the review
update.

## API surfaces

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | Reports API/model readiness and calibration availability. |
| `POST /api/analyze` | Validates and analyzes one fundus image; returns result and encoded artifacts. |
| `POST /api/screening-history` | Creates an anonymous screening-result record. |
| `GET /api/screening-history/{anonymous_patient_id}` | Returns the screening history for an anonymous ID. |
| `PUT /api/screening-history/{record_id}/review` | Adds/updates one of the allowed reviewer decisions and review timestamp. |

There is no authentication or authorization. Local CORS defaults are for
development. Do not expose this service or use it with identifiable patient
data.

## Simulation relationship and integration boundary

The resource simulation is **not invoked by the screening application** and
does not read actual patient records, predictions, queue events, or network
traffic. The Python simulator and Simulink builder take scenario JSON
parameters representing arrivals, stage capacity/service time, transmission,
referable/ungradable fractions, reviewer resources, and outcome assumptions.

The benchmark is an explicit manual handoff:

1. Use an authorized local test image with
   `simulation/benchmark_prototype.py`.
2. If a real checkpoint is present, use its measured median prediction time
   (which includes preprocessing) for `ai_processing_time_seconds`.
3. Use measured quality-assessment time for
   `image_quality_assessment_time_seconds` when its image/environment are
   representative enough for the experiment, recording the conditions.
4. Edit/copy a scenario JSON and run either simulator. Document all other
   assumptions separately.

Current scenario AI latency is a placeholder because no trained model
checkpoint is included. Measured local quality/preprocessing times are from a
generated non-retinal test image and are not representative healthcare or
deployment benchmarks. The model omits some work such as Grad-CAM, retinal
evidence, report rendering, and operational delays. Do not conflate simulator
outputs with observed service performance or demonstrated annual capacity.

## Important failure and trust boundaries

- Missing/unloadable checkpoints are shown as unavailable; the API rejects
  analysis rather than generating a grade without a model.
- Invalid, unsupported, oversized, or corrupt uploads are rejected with
  explicit HTTP errors. Unexpected failures are logged server-side and
  returned as a generic error.
- Image-quality assessment is heuristic; it does not gate out all predictions
  or replace clinical review.
- History identifiers are pseudonymous IDs, not authentication credentials.
  The local database is ignored by Git but remains sensitive local data.
- A clinician remains responsible for the final clinical interpretation.
  Trends are screening-result history only and do not confirm disease
  progression.

## Integration verification

The automated API suite covers upload validation, unavailable/unloadable
models, mocked successful analysis/artifact response, and sanitized failure
handling. Screening-history tests cover local persistence, field validation,
review decisions, and lookup. The frontend has a production build check but no
automated browser-level workflow test. The Python simulation is unit tested;
the Simulink `.slx` cannot be built or run unless MATLAB/Simulink is installed.
See [TESTING.md](../TESTING.md) for the last recorded test results and
limitations.
