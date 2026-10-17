"""ESCO (European Skills, Competences, Qualifications and Occupations) dataset loader.

Expected Schema & File Format:
Source: https://esco.ec.europa.eu/en/use-esco/download
Files:
  - "skills_en.csv": Standard skill concepts definition
  - "occupations_en.csv": Standard occupation concepts definition
  - "occupationSkillRelations_en.csv": Skill to occupation associations

Tidy Target Schema for Occupation-Skill relations:
  - 'esco_occupation_uri': Reference URI of the occupation
  - 'occupation_title': Human-readable occupation title
  - 'skill_name': Standard preferred label of the skill
  - 'relation_type': 'essential' or 'optional'
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


class ESCOLoaderError(Exception):
    """Base exception for ESCO dataset loading failures."""
    pass


class ESCOFileNotFoundError(ESCOLoaderError, FileNotFoundError):
    """Raised when the expected ESCO file does not exist on disk."""
    pass


class ESCOInvalidFormatError(ESCOLoaderError, ValueError):
    """Raised when the ESCO file does not contain expected columns."""
    pass


# Documented expected columns in official ESCO skills_en.csv export
EXPECTED_ESCO_SKILLS_COLUMNS = [
    "conceptType",
    "conceptUri",
    "skillType",
    "reuseLevel",
    "preferredLabel",
    "altLabels",
    "description",
]

# Documented expected columns in official ESCO occupations_en.csv export
EXPECTED_ESCO_OCCUPATIONS_COLUMNS = [
    "conceptUri",
    "preferredLabel",
]

# Documented expected columns in official ESCO occupationSkillRelations_en.csv export
EXPECTED_ESCO_OCC_SKILL_RELATIONS_COLUMNS = [
    "occupationUri",
    "relationType",
    "skillType",
    "skillUri",
]


def load_esco_skills(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate ESCO skills dataset (typically skills_en.csv).

    Parameters
    ----------
    path : Union[str, Path]
        Path to the raw ESCO skills CSV export.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe containing ESCO skill definitions.

    Raises
    ------
    ESCOFileNotFoundError
        If the file at path does not exist.
    ESCOInvalidFormatError
        If required columns are missing or the file is improperly formatted.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ESCOFileNotFoundError(
            f"ESCO Skills file not found at: '{file_path.resolve()}'. "
            "Please ensure raw ESCO CSV exports are downloaded and placed into data/raw/."
        )

    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        raise ESCOInvalidFormatError(f"Failed to parse ESCO Skills file: {e}") from e

    df.columns = [c.strip() for c in df.columns]

    missing_cols = [col for col in EXPECTED_ESCO_SKILLS_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ESCOInvalidFormatError(
            f"ESCO Skills file missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    return df[EXPECTED_ESCO_SKILLS_COLUMNS].copy()


def load_esco_occupations(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate ESCO occupations dataset (typically occupations_en.csv).

    Parameters
    ----------
    path : Union[str, Path]
        Path to the raw ESCO occupations CSV export.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['conceptUri', 'preferredLabel', 'description', 'iscoGroup'].

    Raises
    ------
    ESCOFileNotFoundError
        If the file does not exist.
    ESCOInvalidFormatError
        If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ESCOFileNotFoundError(
            f"ESCO Occupations file not found at: '{file_path.resolve()}'. "
            "Please ensure raw ESCO CSV exports are downloaded and placed into data/raw/."
        )

    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        raise ESCOInvalidFormatError(f"Failed to parse ESCO Occupations file: {e}") from e

    df.columns = [c.strip() for c in df.columns]

    missing_cols = [col for col in EXPECTED_ESCO_OCCUPATIONS_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ESCOInvalidFormatError(
            f"ESCO Occupations file missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    cols = ["conceptUri", "preferredLabel"]
    if "description" in df.columns:
        cols.append("description")
    if "iscoGroup" in df.columns:
        cols.append("iscoGroup")

    return df[cols].copy()


def load_esco_occupation_skill_relations(
    path: Union[str, Path],
    tidy: bool = False,
    skills_path: Optional[Union[str, Path]] = None,
    occupations_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and validate ESCO occupation-skill relations (occupationSkillRelations_en.csv).

    Maintains backwards compatibility with M1 test signature:
    - When `tidy=False`: returns dataframe with raw expected columns:
      ['occupationUri', 'relationType', 'skillType', 'skillUri'].
    - When `tidy=True`: joins human-readable occupation and skill labels to return:
      ['esco_occupation_uri', 'occupation_title', 'skill_name', 'relation_type'].

    Parameters
    ----------
    path : Union[str, Path]
        Path to the raw ESCO occupationSkillRelations CSV export.
    tidy : bool, default False
        Whether to return a tidy DataFrame matching the target shape.
    skills_path : Optional[Union[str, Path]], default None
        Path to skills_en.csv to resolve skillUri to preferredLabel. If None,
        uses 'skillLabel' column if already in the CSV, or checks sibling file.
    occupations_path : Optional[Union[str, Path]], default None
        Path to occupations_en.csv to resolve occupationUri to preferredLabel. If None,
        uses 'occupationLabel' column if present, or checks sibling file.

    Returns
    -------
    pd.DataFrame
        Cleaned or tidy DataFrame containing occupation-skill links.

    Raises
    ------
    ESCOFileNotFoundError
        If the file does not exist.
    ESCOInvalidFormatError
        If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ESCOFileNotFoundError(
            f"ESCO Relations file not found at: '{file_path.resolve()}'. "
            "Please ensure raw ESCO CSV exports are downloaded and placed into data/raw/."
        )

    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        raise ESCOInvalidFormatError(f"Failed to parse ESCO Relations file: {e}") from e

    df.columns = [c.strip() for c in df.columns]

    missing_cols = [col for col in EXPECTED_ESCO_OCC_SKILL_RELATIONS_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ESCOInvalidFormatError(
            f"ESCO Relations file missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    if not tidy:
        return df[EXPECTED_ESCO_OCC_SKILL_RELATIONS_COLUMNS].copy()

    # If occupationLabel is already present in raw CSV export (standard in v1.1+):
    if "occupationLabel" in df.columns:
        occ_titles = df["occupationLabel"]
    else:
        occ_target = occupations_path or (file_path.parent / "occupations_en.csv")
        if Path(occ_target).is_file():
            occ_df = load_esco_occupations(occ_target)
            merged_occ = df.merge(
                occ_df[["conceptUri", "preferredLabel"]],
                left_on="occupationUri",
                right_on="conceptUri",
                how="left",
            )
            occ_titles = merged_occ["preferredLabel"].fillna(df["occupationUri"])
        else:
            occ_titles = df["occupationUri"]

    # If skillLabel is already present in raw CSV export:
    if "skillLabel" in df.columns:
        skill_names = df["skillLabel"]
    else:
        skill_target = skills_path or (file_path.parent / "skills_en.csv")
        if Path(skill_target).is_file():
            skill_df = load_esco_skills(skill_target)
            merged_skill = df.merge(
                skill_df[["conceptUri", "preferredLabel"]],
                left_on="skillUri",
                right_on="conceptUri",
                how="left",
            )
            skill_names = merged_skill["preferredLabel"].fillna(df["skillUri"])
        else:
            skill_names = df["skillUri"]

    tidy_df = pd.DataFrame({
        "esco_occupation_uri": df["occupationUri"].astype(str).str.strip(),
        "occupation_title": occ_titles.astype(str).str.strip(),
        "skill_name": skill_names.astype(str).str.strip(),
        "relation_type": df["relationType"].astype(str).str.strip().str.lower(),
    })

    return tidy_df.drop_duplicates()
