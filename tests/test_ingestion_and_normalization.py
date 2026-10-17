"""Unit and property tests for M1 and M2 normalization and loader compatibility."""

from pathlib import Path
import pandas as pd
import pytest

from src.preprocessing.normalize_skills import (
    normalize_skill_name,
    reconcile_onet_esco,
    FUZZY_MATCH_THRESHOLD,
)
from src.ingestion.onet_loader import (
    load_onet_skills,
    load_onet_occupations,
    ONETFileNotFoundError,
    ONETInvalidFormatError,
)
from src.ingestion.esco_loader import (
    load_esco_skills,
    load_esco_occupations,
    load_esco_occupation_skill_relations,
    ESCOFileNotFoundError,
    ESCOInvalidFormatError,
)


class TestSkillNormalization:
    """Tests for skill string cleaning and standardization."""

    def test_normalize_basic_casing_and_whitespace(self):
        raw = "   Critical   Thinking   "
        expected = "critical thinking"
        assert normalize_skill_name(raw) == expected

    def test_normalize_punctuation_and_delimiters(self):
        raw = "Data-Structures / Algorithms & CI/CD!"
        expected = "data structures algorithms ci cd"
        assert normalize_skill_name(raw) == expected

    def test_normalize_non_string_input(self):
        assert normalize_skill_name(None) == ""
        assert normalize_skill_name(123) == ""

    def test_reconcile_exact_and_fuzzy_synthetic(self):
        # Synthetic test data to verify matching behavior without needing raw files
        onet_df = pd.DataFrame({"skill_name": ["Mathematics", "Time Management", "Quantum Teleportation"]})
        esco_df = pd.DataFrame({"skill_name": ["mathematics", "manage time", "underwater basketweaving"]})

        crosswalk = reconcile_onet_esco(onet_df, esco_df, threshold=80.0)

        assert isinstance(crosswalk, pd.DataFrame)
        assert list(crosswalk.columns) == ["onet_skill_name", "esco_skill_name", "match_type", "match_score"]

        # Exact match check
        exact_row = crosswalk[crosswalk["onet_skill_name"] == "Mathematics"].iloc[0]
        assert exact_row["match_type"] == "exact"
        assert exact_row["match_score"] == 100.0

        # Fuzzy match check ("Time Management" vs "manage time")
        fuzzy_row = crosswalk[crosswalk["onet_skill_name"] == "Time Management"].iloc[0]
        assert fuzzy_row["match_type"] == "fuzzy"
        assert fuzzy_row["match_score"] >= 80.0

        # Unmatched check (unmatched skills kept, not dropped)
        unmatched_onet = crosswalk[crosswalk["onet_skill_name"] == "Quantum Teleportation"].iloc[0]
        assert unmatched_onet["match_type"] == "unmatched"
        assert unmatched_onet["match_score"] == 0.0

        unmatched_esco = crosswalk[crosswalk["esco_skill_name"] == "underwater basketweaving"].iloc[0]
        assert unmatched_esco["match_type"] == "unmatched"
        assert pd.isna(unmatched_esco["onet_skill_name"])


class TestONETLoaderErrorHandling:
    """Tests for O*NET loader without requiring actual dataset files."""

    def test_missing_file_raises_typed_exception(self, tmp_path: Path):
        non_existent_file = tmp_path / "Skills.txt"
        with pytest.raises(ONETFileNotFoundError) as exc_info:
            load_onet_skills(non_existent_file)
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_schema_raises_format_error(self, tmp_path: Path):
        bad_file = tmp_path / "BadSkills.txt"
        bad_file.write_text("ColA\tColB\n1\t2\n")
        with pytest.raises(ONETInvalidFormatError) as exc_info:
            load_onet_skills(bad_file)
        assert "missing expected columns" in str(exc_info.value).lower()

    def test_missing_occupations_file_raises_typed_exception(self, tmp_path: Path):
        non_existent = tmp_path / "Occupation Data.txt"
        with pytest.raises(ONETFileNotFoundError):
            load_onet_occupations(non_existent)


class TestESCOLoaderErrorHandling:
    """Tests for ESCO loader without requiring actual dataset files."""

    def test_missing_skills_file_raises_typed_exception(self, tmp_path: Path):
        non_existent_file = tmp_path / "skills_en.csv"
        with pytest.raises(ESCOFileNotFoundError) as exc_info:
            load_esco_skills(non_existent_file)
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_schema_raises_format_error(self, tmp_path: Path):
        bad_file = tmp_path / "bad_esco.csv"
        bad_file.write_text("random_col_1,random_col_2\nfoo,bar\n")
        with pytest.raises(ESCOInvalidFormatError) as exc_info:
            load_esco_skills(bad_file)
        assert "missing expected columns" in str(exc_info.value).lower()

    def test_missing_relations_file_raises_typed_exception(self, tmp_path: Path):
        non_existent_file = tmp_path / "occupationSkillRelations_en.csv"
        with pytest.raises(ESCOFileNotFoundError):
            load_esco_occupation_skill_relations(non_existent_file)

    def test_missing_occupations_file_raises_typed_exception(self, tmp_path: Path):
        non_existent = tmp_path / "occupations_en.csv"
        with pytest.raises(ESCOFileNotFoundError):
            load_esco_occupations(non_existent)
