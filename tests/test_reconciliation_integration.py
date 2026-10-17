"""Integration tests running against real raw datasets in data/raw/.

Marked with @pytest.mark.integration and skipped automatically if
raw dataset files are not present on disk.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.ingestion.onet_loader import load_onet_skills, load_onet_occupations
from src.ingestion.esco_loader import load_esco_skills, load_esco_occupations, load_esco_occupation_skill_relations
from src.preprocessing.normalize_skills import reconcile_onet_esco, FUZZY_MATCH_THRESHOLD

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ONET_SKILLS = RAW_DIR / "Skills.txt"
ONET_OCC = RAW_DIR / "Occupation Data.txt"
ESCO_SKILLS = RAW_DIR / "skills_en.csv"
ESCO_OCC = RAW_DIR / "occupations_en.csv"
ESCO_RELATIONS = RAW_DIR / "occupationSkillRelations_en.csv"

# Check if real data files exist
HAS_REAL_DATA = (
    ONET_SKILLS.is_file()
    and ONET_OCC.is_file()
    and ESCO_SKILLS.is_file()
    and ESCO_OCC.is_file()
    and ESCO_RELATIONS.is_file()
)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_REAL_DATA, reason="Real raw O*NET and ESCO datasets not present in data/raw/")
class TestRealDataIntegration:
    """End-to-end integration tests using actual O*NET and ESCO datasets."""

    def test_load_real_onet_skills_and_occupations(self):
        # Raw schema load
        raw_df = load_onet_skills(ONET_SKILLS)
        assert len(raw_df) > 0
        assert "O*NET-SOC Code" in raw_df.columns
        assert "Data Value" in raw_df.columns

        # Tidy schema load
        tidy_df = load_onet_skills(ONET_SKILLS, tidy=True, scale_id="IM", occupations_path=ONET_OCC)
        assert len(tidy_df) > 0
        expected_cols = ["onet_soc_code", "occupation_title", "skill_name", "importance_score"]
        assert list(tidy_df.columns) == expected_cols
        assert tidy_df["importance_score"].dtype in ["float64", "float32"]

        # Occupation loader
        occ_df = load_onet_occupations(ONET_OCC)
        assert len(occ_df) > 500
        assert "onet_soc_code" in occ_df.columns

    def test_load_real_esco_skills_and_relations(self):
        # Skills load
        skills_df = load_esco_skills(ESCO_SKILLS)
        assert len(skills_df) > 10000
        assert "preferredLabel" in skills_df.columns

        # Occupations load
        occ_df = load_esco_occupations(ESCO_OCC)
        assert len(occ_df) > 2000
        assert "preferredLabel" in occ_df.columns

        # Tidy relations load
        tidy_rel = load_esco_occupation_skill_relations(ESCO_RELATIONS, tidy=True)
        assert len(tidy_rel) > 50000
        expected_cols = ["esco_occupation_uri", "occupation_title", "skill_name", "relation_type"]
        assert list(tidy_rel.columns) == expected_cols

    def test_reconcile_real_taxonomies(self, tmp_path: Path):
        onet_tidy = load_onet_skills(ONET_SKILLS, tidy=True, scale_id="IM", occupations_path=ONET_OCC)
        esco_tidy = load_esco_occupation_skill_relations(ESCO_RELATIONS, tidy=True)

        report_file = tmp_path / "test_report.md"
        crosswalk = reconcile_onet_esco(
            onet_tidy,
            esco_tidy,
            threshold=FUZZY_MATCH_THRESHOLD,
            report_path=report_file,
        )

        assert isinstance(crosswalk, pd.DataFrame)
        assert len(crosswalk) > 10000
        assert "exact" in crosswalk["match_type"].values
        assert "fuzzy" in crosswalk["match_type"].values
        assert "unmatched" in crosswalk["match_type"].values
        assert report_file.is_file()
