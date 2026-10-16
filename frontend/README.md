# RetinaCare web application

This React/Vite client talks to the Python FastAPI service; it does not contain
prediction values or run the model in the browser. The backend loads the
configured trained checkpoint and returns the actual result from the existing
quality, preprocessing, classifier, calibration, Grad-CAM, retinal-analysis,
and report modules.

## Start the backend

From the repository root, activate the Python environment with the project
dependencies installed, then configure a trained checkpoint:

```powershell
$env:DR_CHECKPOINT = "models/checkpoints/dr_resnet18/best_model.pt"
# Optional: temperature fitted on the validation split for this same checkpoint.
$env:DR_CALIBRATION_TEMPERATURE = "1.2"
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Omit `DR_CALIBRATION_TEMPERATURE` if no fitted calibration temperature is
available. The health endpoint reports when a checkpoint is missing or failed
to load. The API accepts JPEG, PNG, TIFF, and BMP uploads up to 15 MB and
25 megapixels. Uploaded images and generated report files are held in a
temporary directory for the request and are not persisted by the service.

Screening-result history is stored separately in a local SQLite database at
`data/screening_history.sqlite3`. Set `DR_HISTORY_DB` before starting the API
to choose another database path. The database contains only a pre-assigned
anonymous patient ID, screening timestamp, predicted DR level, referable
status, raw model confidence, image-quality score, and optional reviewer
decision. It does not contain images, names, contact details, or demographics.
Use anonymous IDs only and protect the local database as sensitive prototype
data.

## Start the frontend

In another terminal:

```powershell
Set-Location frontend
npm ci
npm run dev
```

Open the local Vite URL (normally `http://localhost:5173`). Configure
`VITE_API_BASE_URL` before starting Vite to use a different API origin, e.g.
`$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"`. The backend allows the
standard local Vite origins by default; configure `DR_CORS_ORIGINS` as a
comma-separated allowlist when using a different origin.

## Workflow and limitations

Upload one retinal fundus image to see engineering image-quality checks,
preprocessed image, model grade and probabilities, referable grouping,
calibrated confidence when configured, Grad-CAM and retinal evidence when
available, and JSON/Markdown/PDF screening reports. Optional visual-evidence
failures are surfaced as unavailable without discarding a successful
classifier result. A qualified reviewer can record one of four outcomes:
AI prediction accepted, AI prediction modified, image requires manual review,
or image ungradable. The outcome is saved with the screening result, anonymous
case ID, screening timestamp, and review timestamp. Use the Screening history
page to look up previous results and view the predicted DR level trend.
Changes between predicted grades are changes in screening results only; they
are not confirmed disease progression.

This prototype does not authenticate reviewers or enforce authorization.
Reviewer controls are a workflow demonstration, not access control. AI output
is screening support only; an ophthalmologist or other qualified clinician
remains responsible for final clinical interpretation. Unsupported or
malformed uploads and unavailable models return explicit errors.

This is an AI-assisted screening research prototype, not a definitive
diagnosis or treatment recommendation. It is not clinically validated;
performance depends on data, labels, and validation conditions. A referable
result recommends ophthalmologist review.

For the complete end-to-end data-flow description, exact demo sequence,
installation commands, and known limitations, see the root
[README.md](../README.md) and [architecture documentation](../docs/ARCHITECTURE.md).
