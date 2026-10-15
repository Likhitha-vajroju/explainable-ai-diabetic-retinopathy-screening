"""Standalone retinal-structure and annotated-evidence research utilities."""

from src.lesion_retinal_analysis.analysis import (
    EVIDENCE_WARNING,
    OpticDiscCandidate,
    evaluate_binary_mask,
    localize_optic_disc,
    save_evidence_visualization,
    segment_vessels,
)

__all__ = [
    "EVIDENCE_WARNING",
    "OpticDiscCandidate",
    "evaluate_binary_mask",
    "localize_optic_disc",
    "save_evidence_visualization",
    "segment_vessels",
]
