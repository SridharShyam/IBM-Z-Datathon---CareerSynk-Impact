"""Unit and integration tests for Stage A conventional ML baseline model (Milestone M4)."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit

from src.models.baseline_relevance import compute_per_occupation_spearman


class TestBaselineModelLogic:
    """Synthetic unit tests for data splitting, metrics, and leakage checks."""

    def test_group_split_prevents_occupation_leakage(self):
        # 10 occupations, 5 skills per occupation = 50 rows
        occupation_ids = np.repeat([f"occ_{i}" for i in range(10)], 5)
        X = np.random.randn(50, 8)
        y = np.random.rand(50)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=occupation_ids))

        train_occs = set(occupation_ids[train_idx])
        test_occs = set(occupation_ids[test_idx])

        # Strict check: intersection must be empty
        assert len(train_occs.intersection(test_occs)) == 0
        assert len(train_occs) == 7
        assert len(test_occs) == 3

    def test_per_occupation_spearman_calculation(self):
        # Perfect positive correlation for occ_1, perfect negative for occ_2
        # occ_1: y_true = [0.2, 0.4, 0.6], y_pred = [1.0, 2.0, 3.0] -> rho = 1.0
        # occ_2: y_true = [0.2, 0.4, 0.6], y_pred = [3.0, 2.0, 1.0] -> rho = -1.0
        y_true = np.array([0.2, 0.4, 0.6, 0.2, 0.4, 0.6])
        y_pred = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
        occ_ids = np.array(["occ_1", "occ_1", "occ_1", "occ_2", "occ_2", "occ_2"])

        mean_rho, med_rho = compute_per_occupation_spearman(y_true, y_pred, occ_ids)
        assert pytest.approx(mean_rho, abs=1e-4) == 0.0
        assert pytest.approx(med_rho, abs=1e-4) == 0.0

        # Perfect ranking across all occupations
        y_pred_perfect = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        mean_rho_perf, med_rho_perf = compute_per_occupation_spearman(y_true, y_pred_perfect, occ_ids)
        assert pytest.approx(mean_rho_perf, abs=1e-4) == 1.0


@pytest.mark.integration
def test_real_baseline_model_artifact_and_metrics():
    """Integration test checking that baseline artifacts and metrics report exist with expected quality."""
    model_path = Path("data/processed/baseline_relevance_model.pkl")
    report_path = Path("data/processed/baseline_metrics.md")

    if not (model_path.is_file() and report_path.is_file()):
        pytest.skip("Baseline model artifacts not present in data/processed/")

    import pickle
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    assert "model" in artifact
    assert "tfidf" in artifact
    assert "skill_emb_map" in artifact

    report_content = report_path.read_text(encoding="utf-8")
    assert "HistGradientBoostingRegressor" in report_content
    assert "Spearman Rank Correlation" in report_content
    assert "DummyRegressor" in report_content
