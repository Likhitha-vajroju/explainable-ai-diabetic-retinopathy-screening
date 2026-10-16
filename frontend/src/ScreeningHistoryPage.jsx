import React, { useEffect, useState } from "react";

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function ScreeningTrend({ records }) {
  const chronological = [...records].sort(
    (first, second) => new Date(first.screened_at) - new Date(second.screened_at),
  );
  const width = 760;
  const height = 230;
  const left = 48;
  const right = width - 20;
  const top = 20;
  const bottom = height - 42;
  const point = (level, index) => ({
    x: chronological.length === 1
      ? (left + right) / 2
      : left + (index / (chronological.length - 1)) * (right - left),
    y: top + ((4 - level) / 4) * (bottom - top),
  });

  return (
    <article className="panel trend-panel">
      <div className="panel-title">
        <span className="panel-icon"><span className="trend-icon">↗</span></span>
        <div><span className="section-kicker">SCREENING-RESULT HISTORY</span><h3>Predicted DR level by screening</h3></div>
      </div>
      <div className="trend-chart">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Chart of predicted DR screening levels over time">
          {[0, 1, 2, 3, 4].map((level) => {
            const y = top + ((4 - level) / 4) * (bottom - top);
            return (
              <g key={level}>
                <line x1={left} x2={right} y1={y} y2={y} className="trend-gridline" />
                <text x={left - 12} y={y + 4} className="trend-axis-label" textAnchor="end">{level}</text>
              </g>
            );
          })}
          {chronological.length > 1 && (
            <polyline
              points={chronological.map((record, index) => {
                const position = point(record.dr_level, index);
                return `${position.x},${position.y}`;
              }).join(" ")}
              className="trend-line"
            />
          )}
          {chronological.map((record, index) => {
            const position = point(record.dr_level, index);
            return (
              <g key={record.id}>
                <circle cx={position.x} cy={position.y} r="6" className="trend-point" />
                <text x={position.x} y={bottom + 25} className="trend-date-label" textAnchor="middle">
                  {new Date(record.screened_at).toLocaleDateString()}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <p className="trend-caveat">
        These are screening results only. Differences between predicted grades are not confirmed disease progression.
      </p>
    </article>
  );
}

function ScreeningHistoryPage({ apiBase, initialPatientId = "" }) {
  const [patientId, setPatientId] = useState(initialPatientId);
  const [records, setRecords] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (initialPatientId) setPatientId(initialPatientId);
  }, [initialPatientId]);

  async function lookupPatient(event) {
    event.preventDefault();
    const normalizedId = patientId.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(normalizedId)) {
      setError("Enter an anonymous ID using letters, numbers, hyphens, or underscores only.");
      setRecords(null);
      return;
    }

    setLoading(true);
    setError("");
    setRecords(null);
    try {
      const response = await fetch(
        `${apiBase}/api/screening-history/${encodeURIComponent(normalizedId)}`,
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(typeof payload.detail === "string" ? payload.detail : "History could not be loaded.");
      }
      setRecords(payload.records);
    } catch (reason) {
      setError(reason.message || "Unable to reach the screening service.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="history-page">
      <div className="history-heading">
        <span className="section-kicker">SCREENING-RESULT HISTORY</span>
        <h1>Previous screening results</h1>
        <p>Look up a record using its anonymous patient ID. No names or contact details are collected.</p>
      </div>

      <form className="panel history-lookup" onSubmit={lookupPatient}>
        <label htmlFor="history-patient-id">Anonymous patient ID</label>
        <div className="history-search-row">
          <input
            id="history-patient-id"
            value={patientId}
            onChange={(event) => setPatientId(event.target.value)}
            placeholder="e.g. ANON-1042"
            maxLength={64}
            autoComplete="off"
          />
          <button className="button button-primary" type="submit" disabled={loading}>
            {loading ? "Searching…" : "Find history"}
          </button>
        </div>
        <p className="form-hint">Use a pre-assigned anonymous ID; do not enter a person's name or other identifying information.</p>
        {error && <div className="inline-error" role="alert">{error}</div>}
      </form>

      {records && records.length === 0 && (
        <div className="panel history-empty" role="status">
          No screening-result history was found for this anonymous ID.
        </div>
      )}

      {records?.length > 0 && (
        <>
          <ScreeningTrend records={records} />
          <article className="panel history-records">
            <div className="panel-title">
              <span className="panel-icon">▤</span>
              <div><span className="section-kicker">SCREENING-RESULT HISTORY</span><h3>{records.length} previous {records.length === 1 ? "result" : "results"}</h3></div>
            </div>
            <div className="history-table-wrap">
              <table className="history-table">
                <thead>
                  <tr>
                    <th scope="col">Screening date/time</th>
                    <th scope="col">DR level</th>
                    <th scope="col">Referable</th>
                    <th scope="col">Confidence</th>
                    <th scope="col">Image quality</th>
                    <th scope="col">Reviewer decision</th>
                  </tr>
                </thead>
                <tbody>
                  {[...records].sort(
                    (first, second) => new Date(second.screened_at) - new Date(first.screened_at),
                  ).map((record) => (
                    <tr key={record.id}>
                      <td>{formatDate(record.screened_at)}</td>
                      <td>Level {record.dr_level}</td>
                      <td>{record.referable ? "Yes" : "No"}</td>
                      <td>{(record.confidence * 100).toFixed(1)}%</td>
                      <td>{record.image_quality_score}/100</td>
                      <td>{record.reviewer_decision?.replaceAll("_", " ") || "Not recorded"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="history-footnote">A screening-result history is not a medical record or a confirmed diagnosis.</p>
          </article>
        </>
      )}
    </section>
  );
}

export default ScreeningHistoryPage;
