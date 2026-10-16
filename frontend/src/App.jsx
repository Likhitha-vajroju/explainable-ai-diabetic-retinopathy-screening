import React, { useEffect, useRef, useState } from "react";
import ScreeningHistoryPage from "./ScreeningHistoryPage.jsx";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const MAX_BYTES = 15 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/tiff", "image/bmp"];

function Icon({ name, size = 20 }) {
  const paths = {
    eye: <><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" /></>,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M4 16v3h16v-3" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    alert: <><path d="M12 3 2.8 19h18.4L12 3Z" /><path d="M12 9v4m0 3h.01" /></>,
    file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" /><path d="M14 2v6h6M8 13h8m-8 4h8" /></>,
    shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z" /><path d="m9 12 2 2 4-4" /></>,
    pulse: <path d="M3 12h4l3-8 4 16 3-8h4" />,
    download: <><path d="M12 3v12m0 0 5-5m-5 5-5-5" /><path d="M4 17v4h16v-4" /></>,
    close: <path d="m18 6-12 12M6 6l12 12" />,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

function fileError(file) {
  if (!file) return "Choose an image to continue.";
  if (!ACCEPTED_TYPES.includes(file.type)) return "Unsupported image. Use JPEG, PNG, TIFF, or BMP.";
  if (file.size > MAX_BYTES) return "Image is larger than the 15 MB upload limit.";
  if (file.size === 0) return "The selected file is empty.";
  return "";
}

function decodeBase64Utf8(value) {
  const binary = atob(value);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

function DownloadButton({ artifact, label, filename, icon = "download" }) {
  if (!artifact?.base64) return null;
  const href = `data:${artifact.media_type};base64,${artifact.base64}`;
  return (
    <a className="button button-secondary button-small" href={href} download={filename}>
      <Icon name={icon} size={16} /> {label}
    </a>
  );
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [artifacts, setArtifacts] = useState({});
  const [error, setError] = useState("");
  const [health, setHealth] = useState({ loading: true });
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showEnhanced, setShowEnhanced] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [page, setPage] = useState("screening");
  const [reviewCaseId, setReviewCaseId] = useState("");
  const [reviewerDecision, setReviewerDecision] = useState("");
  const [reviewRecord, setReviewRecord] = useState(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const [reviewedAt, setReviewedAt] = useState("");

  async function checkServiceHealth(isCurrent = () => true) {
    if (isCurrent()) setHealth({ loading: true });
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      if (!response.ok) throw new Error("The screening API is not responding.");
      const data = await response.json();
      if (isCurrent()) setHealth({ ...data, loading: false });
    } catch (reason) {
      if (isCurrent()) {
        setHealth({
          loading: false,
          model_ready: false,
          model_error: reason instanceof Error
            ? reason.message
            : "The screening API is not responding.",
        });
      }
    }
  }

  useEffect(() => {
    let active = true;
    checkServiceHealth(() => active);
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function selectFile(candidate) {
    if (!candidate) return;
    setResult(null);
    setArtifacts({});
    setReviewRecord(null);
    setReviewedAt("");
    setReviewerDecision("");
    setShowEnhanced(false);
    const validationError = fileError(candidate);
    if (validationError) {
      setFile(null);
      setError(validationError);
      return;
    }
    setError("");
    setFile(candidate);
  }

  async function analyze() {
    if (!file || fileError(file)) {
      setError(fileError(file));
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    setArtifacts({});
    setReviewError("");
    setReviewRecord(null);
    setReviewedAt("");
    setReviewerDecision("");
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 120_000);
    try {
      const formData = new FormData();
      formData.append("image", file);
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "Analysis could not be completed.";
        throw new Error(detail);
      }
      setResult(payload.result);
      setArtifacts(payload.artifacts || {});
    } catch (reason) {
      setError(
        reason instanceof Error && reason.name === "AbortError"
          ? "Analysis timed out after two minutes. Check the service and try again."
          : reason.message || "Unable to reach the screening service. Check that the backend is running.",
      );
    } finally {
      window.clearTimeout(timeout);
      setLoading(false);
    }
  }

  async function submitReviewerDecision(event) {
    event.preventDefault();
    const anonymousPatientId = reviewCaseId.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(anonymousPatientId)) {
      setReviewError("Enter an anonymous case ID using letters, numbers, hyphens, or underscores only.");
      return;
    }
    if (!reviewerDecision) {
      setReviewError("Select a reviewer decision before submitting this case.");
      return;
    }

    setReviewLoading(true);
    setReviewError("");
    try {
      let record = reviewRecord;
      if (!record) {
        const createResponse = await fetch(`${API_BASE}/api/screening-history`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            anonymous_patient_id: anonymousPatientId,
            dr_level: prediction.grade,
            referable: screening.referable,
            confidence: prediction.raw_confidence,
            image_quality_score: quality.score,
          }),
        });
        const createPayload = await createResponse.json().catch(() => ({}));
        if (!createResponse.ok) {
          throw new Error(typeof createPayload.detail === "string" ? createPayload.detail : "Screening case could not be saved.");
        }
        record = createPayload;
        setReviewRecord(record);
      }

      const reviewResponse = await fetch(`${API_BASE}/api/screening-history/${record.id}/review`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer_decision: reviewerDecision }),
      });
      const reviewPayload = await reviewResponse.json().catch(() => ({}));
      if (!reviewResponse.ok) {
        throw new Error(typeof reviewPayload.detail === "string" ? reviewPayload.detail : "Reviewer decision could not be saved.");
      }
      setReviewRecord(reviewPayload);
      setReviewedAt(reviewPayload.reviewed_at);
    } catch (reason) {
      setReviewError(reason.message || "Unable to save the reviewer decision.");
    } finally {
      setReviewLoading(false);
    }
  }

  const screening = result?.referable_screening;
  const prediction = result?.dr_prediction;
  const quality = result?.image_quality;
  const gradcam = result?.gradcam;
  const retinal = result?.retinal_lesion_evidence;
  const recommendations = screening?.recommendation || "";
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="RetinaCare home">
          <span className="brand-mark"><Icon name="eye" size={23} /></span>
          <span><strong>RetinaCare</strong><small>SCREENING RESEARCH PROTOTYPE</small></span>
        </a>
        <div className="topbar-right">
          <nav className="main-nav" aria-label="Main navigation">
            <button className={page === "screening" ? "nav-link active" : "nav-link"} onClick={() => setPage("screening")}>New screening</button>
            <button className={page === "history" ? "nav-link active" : "nav-link"} onClick={() => setPage("history")}>Screening history</button>
          </nav>
          <span className={`service-status ${health.model_ready ? "online" : "offline"}`}>
            <span className="status-dot" />
            {health.loading ? "Connecting to service" : health.model_ready ? "Analysis service ready" : "Service unavailable"}
          </span>
          <span className="topbar-shield"><Icon name="shield" size={17} /> Research use</span>
        </div>
      </header>

      <main>
        {page === "screening" && <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow"><span className="eyebrow-line" /> DIABETIC RETINOPATHY · IMAGE REVIEW</p>
            <h1>Screen with clarity.<br /><span>Review with confidence.</span></h1>
            <p className="hero-description">
              Explore image quality, an experimental DR grade, and visual model evidence in one review-ready workspace.
            </p>
            <div className="prototype-banner">
              <Icon name="alert" size={18} />
              <span><strong>AI-assisted screening — not a definitive diagnosis.</strong> Outputs need qualified clinical review.</span>
            </div>
          </div>
          <div className="hero-art" aria-hidden="true">
            <div className="orbit orbit-one" /><div className="orbit orbit-two" />
            <div className="retina-disc"><div className="retina-core" /><i /><i /><i /><i /><i /><i /></div>
            <div className="art-label"><span className="status-dot" /> IMAGE REVIEW WORKSPACE</div>
          </div>
        </section>}

        {page === "screening" && <section className="workflow-heading">
          <div><span className="section-kicker">SCREENING WORKFLOW</span><h2>New image review</h2></div>
          <p>Results are generated by the connected research model—not sample values.</p>
        </section>}

        {page === "screening" && !result && (
          <section className="upload-card">
            <div className="card-heading">
              <div className="step-number">01</div>
              <div><h3>Upload a fundus image</h3><p>Use a clear retinal photograph in JPEG, PNG, TIFF, or BMP format.</p></div>
            </div>
            <div
              className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]); }}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => { if (!file && (event.key === "Enter" || event.key === " ")) inputRef.current?.click(); }}
              role={file ? "group" : "button"}
              tabIndex={file ? undefined : 0}
              aria-label="Choose a fundus image"
            >
              <input
                ref={inputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,image/jpeg,image/png,image/tiff,image/bmp"
                onChange={(event) => selectFile(event.target.files?.[0])}
                hidden
              />
              {file ? (
                <div className="selected-file">
                  <img src={preview} alt="Selected fundus image preview" />
                  <div className="selected-file-details"><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to analyze</span></div>
                  <button className="icon-button" type="button" aria-label="Remove selected image" onClick={(event) => { event.stopPropagation(); setFile(null); setError(""); if (inputRef.current) inputRef.current.value = ""; }}><Icon name="close" /></button>
                </div>
              ) : (
                <><span className="upload-icon"><Icon name="upload" size={25} /></span><strong>Drop your fundus image here</strong><span className="drop-subtitle">or <u>browse files</u> on your device</span><span className="format-note">JPEG · PNG · TIFF · BMP <b>·</b> Up to 15 MB</span></>
              )}
            </div>
            {error && <div className="inline-error" role="alert"><Icon name="alert" size={17} />{error}</div>}
            {!health.loading && !health.model_ready && <div className="service-notice" role="status"><Icon name="alert" size={17} /><span><strong>Model not ready.</strong> {health.model_error || "Start the API with a configured trained checkpoint before analyzing."} <button className="text-button" type="button" onClick={() => checkServiceHealth()}>Check again</button></span></div>}
            <div className="upload-actions">
              <span className="privacy-note"><Icon name="shield" size={16} /> Image processed for this request; not stored by the API.</span>
              <button className="button button-primary" onClick={analyze} disabled={!file || loading || !health.model_ready}>
                {loading ? <><span className="spinner" /> Analyzing image…</> : <>Analyze image <span aria-hidden="true">→</span></>}
              </button>
            </div>
            {loading && <div className="loading-note" role="status"><span className="spinner spinner-dark" /> Assessing image quality, running the DR model, and preparing visual evidence. This can take a moment.</div>}
          </section>
        )}

        {page === "screening" && result && (
          <section className="results">
            <div className="result-toolbar">
              <div><span className="section-kicker">IMAGE REVIEW COMPLETE</span><h2>Screening summary</h2></div>
              <div className="toolbar-actions">
                <button className="button button-secondary" onClick={() => { setResult(null); setArtifacts({}); setShowEnhanced(false); }}>Review another image</button>
                <DownloadButton artifact={artifacts.screening_report_pdf} label="Download report" filename="screening-report.pdf" />
              </div>
            </div>
            <div className="disclaimer-strip"><Icon name="shield" size={18} /><span><strong>AI-assisted screening — not a definitive diagnosis.</strong> Performance depends on dataset and validation conditions.</span></div>
            <div className={`referral-banner ${screening?.referable ? "referable" : "nonreferable"}`}>
              <div className="referral-icon"><Icon name={screening?.referable ? "alert" : "check"} size={22} /></div>
              <div><span className="banner-label">{screening?.referable ? "REFERABLE SCREENING RESULT" : "NON-REFERABLE UNDER PROJECT DEFINITION"}</span><h3>{recommendations}</h3><p>Screening grouping: Levels 0–1 non-referable · Levels 2–4 referable</p></div>
            </div>

            <div className="summary-grid">
              <article className="panel grade-panel">
                <div className="panel-title"><span className="panel-icon"><Icon name="pulse" /></span><div><span className="section-kicker">MODEL OUTPUT</span><h3>Predicted DR grade</h3></div></div>
                <div className="grade-display"><span className="grade-number">{prediction.grade}</span><div><span className="grade-label">DR LEVEL {prediction.grade}</span><strong>{prediction.grade_label}</strong></div></div>
                <div className="confidence-row"><span>Raw class probability</span><strong>{(prediction.raw_confidence * 100).toFixed(1)}%</strong></div>
                <div className="confidence-row"><span>Calibrated confidence</span><strong>{prediction.calibrated_confidence == null ? "Not available" : `${(prediction.calibrated_confidence * 100).toFixed(1)}%`}</strong></div>
                <p className="microcopy">{prediction.confidence_note}</p>
                <div className="probability-list">
                  {prediction.raw_probabilities.map((item) => (
                    <div className="probability-item" key={item.dr_level}>
                      <div className="probability-label"><span>Level {item.dr_level} · {item.label}</span><strong>{(item.probability * 100).toFixed(1)}%</strong></div>
                      <div className="bar-track"><span className={item.dr_level === prediction.grade ? "bar-fill selected" : "bar-fill"} style={{ width: `${Math.max(0, Math.min(100, item.probability * 100))}%` }} /></div>
                    </div>
                  ))}
                </div>
              </article>

              <article className="panel quality-panel">
                <div className="panel-title"><span className="panel-icon"><Icon name="eye" /></span><div><span className="section-kicker">IMAGE SUITABILITY</span><h3>Image quality</h3></div></div>
                <div className="quality-score"><strong>{quality.score}</strong><span>/ 100</span><span className={`quality-badge ${quality.status.toLowerCase()}`}>{quality.status}</span></div>
                <p className="microcopy">Engineering quality checks; not a clinically validated measure of gradability.</p>
                <div className="measurement-list">
                  {quality.measurements.map((metric) => (
                    <div className="measurement-row" key={metric.name}><span>{metric.name.replaceAll("_", " ")}</span><strong className={`metric-status ${metric.status.toLowerCase()}`}>{metric.status}</strong></div>
                  ))}
                </div>
                {quality.reasons?.length > 0 && <div className="quality-reasons"><strong>Review flags</strong><ul>{quality.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>}
                <div className="image-dimensions">{result.image.width} × {result.image.height} px · {result.image.format || "Image"}</div>
              </article>
            </div>

            <article className="panel evidence-panel">
              <div className="evidence-heading"><div className="panel-title"><span className="panel-icon"><Icon name="eye" /></span><div><span className="section-kicker">VISUAL REVIEW</span><h3>Image and model evidence</h3></div></div><p>Attention maps and algorithmic candidates are not confirmed lesions.</p></div>
              <div className="evidence-grid">
                <div className="evidence-image-card">
                  <div className="evidence-image">
                    {showEnhanced && artifacts.preprocessed_image ? <img src={artifacts.preprocessed_image.data_url} alt="Preprocessed fundus image" /> : <img src={preview} alt="Original uploaded fundus image" />}
                  </div>
                  <div className="evidence-image-caption"><span>{showEnhanced ? "Preprocessed image" : "Original uploaded image"}</span>{artifacts.preprocessed_image && <button className="text-button" onClick={() => setShowEnhanced(!showEnhanced)}>{showEnhanced ? "Show original" : "View preprocessing"}</button>}</div>
                </div>
                {gradcam?.status === "available" && artifacts.gradcam ? (
                  <div className="evidence-image-card">
                    <div className="evidence-image"><img src={artifacts.gradcam.data_url} alt="Grad-CAM model attention visualization with original, heatmap and overlay" /></div>
                    <div className="evidence-image-caption"><span>Grad-CAM · model attention</span><span className="evidence-available"><Icon name="check" size={14} /> Available</span></div>
                    <p className="image-note">{gradcam.warning}</p>
                  </div>
                ) : (
                  <div className="evidence-unavailable"><Icon name="alert" /><strong>Grad-CAM unavailable</strong><span>{gradcam?.message || "No attention visualization was returned by the backend."}</span></div>
                )}
                {retinal?.status === "available" && artifacts.retinal_evidence ? (
                  <div className="evidence-image-card">
                    <div className="evidence-image"><img src={artifacts.retinal_evidence.data_url} alt="Algorithmic retinal structure candidates and any annotated evidence" /></div>
                    <div className="evidence-image-caption"><span>Retinal structure candidates</span><span className="evidence-available"><Icon name="check" size={14} /> Available</span></div>
                    <p className="image-note">{retinal.warning} {retinal.lesion_annotations}.</p>
                  </div>
                ) : (
                  <div className="evidence-unavailable"><Icon name="alert" /><strong>Retinal evidence unavailable</strong><span>{retinal?.message || "No retinal-analysis visualization was returned."}</span></div>
                )}
              </div>
              <div className="preprocessing-note"><Icon name="check" size={16} /><span><strong>Preprocessing complete.</strong> {result.preprocessing.description}</span></div>
            </article>

            <div className="next-step-card">
              <div className="next-icon"><Icon name="pulse" size={21} /></div>
              <div><span className="section-kicker">RECOMMENDED NEXT ACTION</span><p>{recommendations}</p></div>
            </div>

            <form className="panel reviewer-workflow" onSubmit={submitReviewerDecision}>
              <div className="reviewer-heading">
                <span className="section-kicker">HUMAN REVIEW REQUIRED</span>
                <h3>Qualified reviewer decision</h3>
                <p>AI outputs are screening support only. An ophthalmologist or other qualified clinician remains responsible for final clinical interpretation.</p>
              </div>
              <div className="reviewer-evidence-summary">
                <span>Review this case with the original fundus image, DR prediction, confidence, referable status, Grad-CAM and any available retinal evidence above.</span>
                <span>Screening recommendation: {recommendations}</span>
              </div>
              <div className="reviewer-fields">
                <label htmlFor="review-case-id">Anonymous case ID</label>
                <input
                  id="review-case-id"
                  value={reviewCaseId}
                  onChange={(event) => setReviewCaseId(event.target.value)}
                  placeholder="e.g. CASE-1042"
                  maxLength={64}
                  autoComplete="off"
                  disabled={Boolean(reviewRecord)}
                  required
                />
                <p className="form-hint">Use an anonymous, pre-assigned ID only. This prototype does not authenticate reviewer identity.</p>
                <label htmlFor="reviewer-decision">Review outcome</label>
                <select
                  id="reviewer-decision"
                  value={reviewerDecision}
                  onChange={(event) => setReviewerDecision(event.target.value)}
                  disabled={Boolean(reviewedAt)}
                  required
                >
                  <option value="" disabled>Select a reviewer decision</option>
                  <option value="ai_prediction_accepted">AI prediction accepted</option>
                  <option value="ai_prediction_modified">AI prediction modified</option>
                  <option value="image_requires_manual_review">Image requires manual review</option>
                  <option value="image_ungradable">Image ungradable</option>
                </select>
              </div>
              {reviewError && <div className="inline-error" role="alert">{reviewError}</div>}
              {reviewedAt && (
                <div className="history-save-success" role="status">
                  <span>Reviewer decision saved at {new Date(reviewedAt).toLocaleString()} for case {reviewRecord.anonymous_patient_id}.</span>
                  <button type="button" className="text-button" onClick={() => setPage("history")}>View screening-result history</button>
                </div>
              )}
              <button className="button button-primary" type="submit" disabled={reviewLoading || Boolean(reviewedAt)}>
                {reviewLoading ? "Saving review…" : reviewedAt ? "Reviewer decision recorded" : reviewRecord ? "Retry saving reviewer decision" : "Record reviewer decision"}
              </button>
              <p className="reviewer-audit-note">Case and screening timestamps are stored locally with this decision. No uploaded image is stored.</p>
            </form>

            <article className="report-card">
              <div className="report-card-copy"><span className="report-icon"><Icon name="file" size={21} /></span><div><h3>Screening report</h3><p>Review or save the generated report and analysis details.</p></div></div>
              <div className="report-actions">
                <button className="button button-secondary button-small" onClick={() => setReportOpen(!reportOpen)}><Icon name="file" size={16} /> {reportOpen ? "Hide report" : "View report"}</button>
                <DownloadButton artifact={artifacts.screening_report_pdf} label="PDF report" filename="screening-report.pdf" />
                <DownloadButton artifact={artifacts.screening_report_markdown} label="Markdown" filename="screening-report.md" />
                <DownloadButton artifact={artifacts.screening_report_json} label="JSON data" filename="screening-result.json" />
              </div>
              {reportOpen && artifacts.screening_report_markdown && <pre className="report-preview">{decodeBase64Utf8(artifacts.screening_report_markdown.base64)}</pre>}
            </article>
          </section>
        )}

        {page === "history" && (
          <ScreeningHistoryPage
            apiBase={API_BASE}
            initialPatientId={reviewRecord?.anonymous_patient_id || ""}
          />
        )}
      </main>

      <footer className="footer">
        <div className="footer-brand"><span className="brand-mark small"><Icon name="eye" size={18} /></span><strong>RetinaCare</strong></div>
        <p>AI-assisted screening prototype. Not a medical device or a substitute for professional clinical assessment.</p>
        <span className="footer-status"><span className="status-dot" /> Research prototype</span>
      </footer>
    </div>
  );
}

export default App;
