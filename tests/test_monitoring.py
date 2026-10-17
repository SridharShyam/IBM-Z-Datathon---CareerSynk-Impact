"""Tests for data quality checks, drift detection, and inference telemetry."""

import pytest
import numpy as np

from src.models.model_monitoring import DataQualityAuditor, DriftMonitor, InferenceTracker


def test_data_quality_auditor_happy_path():
    """Verify clean inputs produce 100% match rate and zero flags."""
    known_skills = {"python", "git", "sql", "docker"}
    known_occs = {"occ:onet:15-1252.00"}
    auditor = DataQualityAuditor(known_skills, known_occs)

    audit = auditor.audit_request(["Python", "SQL"], "occ:onet:15-1252.00")
    assert audit["valid"] is True
    assert audit["vocabulary_coverage_rate"] == 1.0
    assert len(audit["quality_flags"]) == 0
    assert audit["unmatched_skills_count"] == 0


def test_data_quality_auditor_flags_anomalies():
    """Verify auditor flags unknown occupations and high OOV rates."""
    known_skills = {"python", "git"}
    known_occs = {"occ:onet:15-1252.00"}
    auditor = DataQualityAuditor(known_skills, known_occs)

    # Empty skills
    audit_empty = auditor.audit_request([], "occ:onet:15-1252.00")
    assert "EMPTY_SKILLS_LIST" in audit_empty["quality_flags"]

    # High OOV skills
    audit_oov = auditor.audit_request(["farming", "welding", "plumbing", "driving"], "occ:onet:15-1252.00")
    assert "HIGH_OUT_OF_VOCABULARY_RATE" in audit_oov["quality_flags"]
    assert audit_oov["vocabulary_coverage_rate"] == 0.0

    # Unknown occupation
    audit_occ = auditor.audit_request(["Python"], "unknown_occ_99")
    assert "UNKNOWN_OCCUPATION_ID" in audit_occ["quality_flags"]


def test_drift_monitor_psi_and_ks():
    """Verify PSI and KS statistics correctly identify stability vs shift."""
    np.random.seed(42)
    baseline = np.random.normal(loc=0.5, scale=0.1, size=500)
    in_dist = np.random.normal(loc=0.5, scale=0.1, size=200)
    shifted = np.random.normal(loc=0.8, scale=0.1, size=200)

    # Stable distribution
    psi_stable = DriftMonitor.calculate_psi(baseline, in_dist)
    assert psi_stable < 0.10

    # Shifted distribution
    psi_drift = DriftMonitor.calculate_psi(baseline, shifted)
    assert psi_drift > 0.25

    # KS test
    ks_stat_in, p_in, drift_in = DriftMonitor.test_distribution_shift_ks(baseline, in_dist)
    assert drift_in is False

    ks_stat_dr, p_dr, drift_dr = DriftMonitor.test_distribution_shift_ks(baseline, shifted)
    assert drift_dr is True
    assert p_dr < 0.01


def test_inference_tracker_logging(tmp_path):
    """Verify telemetry logging writes valid JSON lines and loads into DataFrame."""
    tracker = InferenceTracker(log_dir=str(tmp_path))
    tracker.log_inference(
        endpoint="/api/v1/readiness",
        occupation_id="occ:onet:15-1252.00",
        candidate_skills_count=4,
        vocabulary_coverage=0.75,
        readiness_score=0.45,
        latency_ms=12.5,
        quality_flags=[],
    )

    df = tracker.load_telemetry_df()
    assert len(df) == 1
    assert df.iloc[0]["endpoint"] == "/api/v1/readiness"
    assert df.iloc[0]["readiness_score"] == 0.45
    assert df.iloc[0]["latency_ms"] == 12.5
