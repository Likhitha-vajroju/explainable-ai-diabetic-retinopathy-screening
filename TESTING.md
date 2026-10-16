# Testing, reproducibility, and configuration

## Automated checks

Run the Python unit and integration suite from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The suite covers preprocessing and image-quality behavior, classification and
evaluation helpers, explainability, reports, screening-history persistence and
validation, API upload/error/readiness behavior, and resource-simulation
invariants. API integration tests use FastAPI's in-process ASGI transport and
mock model/report generation where a trained checkpoint is not available.

Build the frontend from its directory:

```powershell
Set-Location frontend
npm run build
```

There is no frontend unit-test runner configured. The production build checks
that the React/Vite application compiles and bundles; it is not a substitute
for browser-level workflow or accessibility testing.

## Results for this engineering pass

| Check | Result |
| --- | --- |
| Python unit/integration suite | **Passed: 78 tests** |
| Frontend production build | **Passed:** `npm run build` from `frontend/` |
| `git diff --check` | Passed |
| Tracked data/model artifact check | No tracked dataset images/CSVs or model checkpoints found |
| Credential and user-specific path scan | No credentials or user-specific absolute paths found in tracked source; path examples in documentation are intentional placeholders |

Tests that deliberately exercise missing or invalid checkpoints and internal
API failures emit expected log messages. Those log entries are part of the
negative-path checks and do not indicate suite failures.

## Reproducibility

- Use the same Python and Node.js versions and dependency sets when comparing
  runs. Python dependencies are listed with minimum versions in
  `requirements.txt`; they are not fully pinned, so fresh environments may
  resolve newer compatible packages. The frontend dependency graph is recorded
  in `frontend/package-lock.json`; use `npm ci` for a lockfile-based install.
- The simulation accepts an explicit random seed and keeps generated arrivals
  consistent when comparing resource scenarios. See
  [`simulation/README.md`](simulation/README.md) for its inputs, assumptions,
  outputs, and limitations.
- The API reads configuration from environment variables; see the section
  below and [`frontend/README.md`](frontend/README.md).
- No predictions, model weights, or evaluation results were changed as part of
  the engineering-quality pass. The API integration tests use generated test
  images and mocked analysis/report outputs, not real model performance data.

## Configuration

| Setting | Purpose |
| --- | --- |
| `DR_CHECKPOINT` | Path to the trained classifier checkpoint. If absent or unloadable, the API reports that the model is unavailable and analysis requests fail explicitly. |
| `DR_CALIBRATION_TEMPERATURE` | Optional finite positive calibration temperature for the configured checkpoint. Leave unset when no fitted value for that checkpoint is available. |
| `DR_HISTORY_DB` | Optional path to the local SQLite screening-history database. Defaults to `data/screening_history.sqlite3`. |
| `DR_CORS_ORIGINS` | Comma-separated API CORS origin allowlist. Defaults to the two local Vite origins documented in `frontend/README.md`. |
| `VITE_API_BASE_URL` | Optional API origin used by the frontend when it differs from its default. |
| `DR_SIMULATION_SCENARIO` | Optional JSON scenario path consumed by the MATLAB/Simulink model builder; see `simulation/README.md`. |

Do not put names, contact information, or other direct identifiers in the
anonymous history ID. The local SQLite database is ignored by Git but should
still be treated as sensitive prototype data.

## Engineering checks and audit

- Upload validation tests include empty, corrupt/truncated, unsupported, and
  valid mocked image requests. The API enforces format, byte-size, and pixel
  limits, reports unavailable/malformed model states, and returns a generic
  error response for unexpected analysis failures while logging details
  server-side.
- History input validation rejects invalid anonymous IDs, DR levels, and
  non-finite/out-of-range measurements. Simulation configuration validates
  integral resource/count parameters.
- The declared Python dependencies have corresponding project usage;
  `httpx` supports the API integration tests. The frontend manifest has no
  separate test framework or unused application dependencies identified in
  this pass.
- The tracked-file check found no dataset contents, trained model artifacts,
  or generated `.slx` files. This worktree does contain an ignored local
  `data/aptos-20152019-preprocessed/` dataset directory and an ignored SQLite
  history database; neither is tracked or included in the changes. Confirm
  dataset provenance and permissions before using local data.
- Credential scans found no hard-coded credentials in tracked source or
  documentation. Absolute-path matches were intentional `C:\path\to\...`
  examples, not machine-specific user paths.
- Documentation and the UI describe outputs as screening results, not
  diagnosis or confirmed disease progression. They state that final clinical
  interpretation remains the responsibility of a qualified clinician.

## Known limitations

- No trained checkpoint is present in this workspace. Consequently, automated
  tests validate API behavior with mocks and do not verify real checkpoint
  loading, predictions, calibration, or measured prediction latency.
- No MATLAB/Simulink executable was available to generate or execute the
  `.slx` model. The Python resource-allocation simulator is covered by tests,
  but the Simulink integration remains unverified here.
- The local history/reviewer prototype has no authentication or authorization.
  It is not suitable for storing identifiable patient data or deployment in a
  clinical environment.
- Frontend behavior has a production-build check but no automated browser
  test coverage. Real-server interaction, visual regressions, and accessibility
  remain to be tested.
- Git does not track dataset contents or model artifacts in this repository.
  Dataset provenance, label interpretation, clinical validity, and performance
  in external populations/devices remain outside the scope of these tests.
