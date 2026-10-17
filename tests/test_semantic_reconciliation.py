"""Unit tests for semantic skill reconciliation and hierarchy boost logic (Milestone M3)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.semantic_reconciliation import (
    compute_skill_embeddings,
    is_hierarchy_supported,
    reconcile_onet_esco_semantic,
    TOP_K_MATCHES,
    MIN_SIMILARITY_THRESHOLD,
)


class TestSemanticReconciliationFixtures:
    """Tests using mock / synthetic fixtures to verify algorithmic properties without model download."""

    def test_hierarchy_support_logic(self):
        mock_hierarchy = {
            "transversal_skills": {"active listening", "think critically", "manage time"},
            "ancestors": {
                "python programming": {"computer programming", "software development"},
                "clinical trial analysis": {"data analysis", "medical research"},
            },
        }

        # Case 1: Skill is in transversal skills set
        assert is_hierarchy_supported("active listening", "active listening", mock_hierarchy) is True

        # Case 2: Ancestor matches keyword in O*NET construct
        assert is_hierarchy_supported("programming", "python programming", mock_hierarchy) is True

        # Case 3: Completely unrelated skill
        assert is_hierarchy_supported("welding", "clinical trial analysis", mock_hierarchy) is False

    def test_empty_or_none_hierarchy_graceful_handling(self):
        assert is_hierarchy_supported("critical thinking", "think critically", None) is False
        assert is_hierarchy_supported("critical thinking", "think critically", {}) is False

    def test_semantic_crosswalk_structure_and_topk(self, monkeypatch):
        # Synthetic mock for SentenceTransformer to run fast unit tests
        class MockModel:
            def encode(self, texts, **kwargs):
                # Return deterministic pseudo-vectors (len(texts), 4)
                arr = []
                for t in texts:
                    val = len(t) % 5 + 1.0
                    arr.append([val, 1.0, 0.5, 0.2])
                normed = np.array(arr, dtype=np.float32)
                norms = np.linalg.norm(normed, axis=1, keepdims=True)
                return normed / norms

        monkeypatch.setattr("src.preprocessing.semantic_reconciliation.SentenceTransformer", lambda *args, **kwargs: MockModel())

        onet_df = pd.DataFrame({"skill_name": ["Programming", "Time Management"]})
        esco_df = pd.DataFrame({"skill_name": [f"esco skill {i}" for i in range(20)]})

        crosswalk_df, stats = reconcile_onet_esco_semantic(
            onet_df=onet_df,
            esco_df=esco_df,
            top_k=5,
            min_similarity=0.1,  # Low threshold to ensure candidates pass in mock
        )

        assert isinstance(crosswalk_df, pd.DataFrame)
        expected_cols = ["onet_skill_name", "esco_skill_name", "embedding_score", "hierarchy_supported", "included_in_topk"]
        assert list(crosswalk_df.columns) == expected_cols

        # Test Top-K constraint per O*NET skill
        for onet_name, group in crosswalk_df.groupby("onet_skill_name"):
            assert len(group) <= 5
            assert (group["included_in_topk"] == True).all()

        assert stats["total_onet_skills"] == 2
        assert stats["total_esco_skills"] == 20


@pytest.mark.integration
def test_semantic_reconciliation_integration_real_data(tmp_path: Path):
    """Integration test running semantic reconciliation on real data files if present."""
    raw_dir = Path("data/raw")
    skills_txt = raw_dir / "Skills.txt"
    esco_csv = raw_dir / "skills_en.csv"
    h_csv = raw_dir / "skillsHierarchy_en.csv"
    t_csv = raw_dir / "transversalSkillsCollection_en.csv"

    if not (skills_txt.is_file() and esco_csv.is_file() and h_csv.is_file() and t_csv.is_file()):
        pytest.skip("Required real data or hierarchy files not present in data/raw/")

    onet_df = pd.DataFrame({"skill_name": ["Time Management", "Speaking", "Operations Analysis"]})
    esco_df = pd.read_csv(esco_csv, nrows=1000)

    report_path = tmp_path / "test_semantic_report.md"
    crosswalk_df, stats = reconcile_onet_esco_semantic(
        onet_df=onet_df,
        esco_df=esco_df,
        hierarchy_path=h_csv,
        transversal_path=t_csv,
        top_k=3,
        min_similarity=0.3,
        report_path=report_path,
    )

    assert isinstance(crosswalk_df, pd.DataFrame)
    assert len(crosswalk_df) > 0
    assert stats["embedding_time_esco_s"] > 0
