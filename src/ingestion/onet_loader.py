"""O*NET dataset loader for Skills and Occupation Data.

Expected Schema & File Format:
Source: https://www.onetcenter.org/database.html
File: "Skills.txt" (Tab-delimited text file or CSV)
Primary Columns Expected:
  - 'O*NET-SOC Code': Unique occupational identifier (e.g. '15-1252.00')
  - 'Element ID': Standardized skill identifier (e.g. '2.A.1.a')
  - 'Element Name': Human-readable name of skill (e.g. 'Reading Comprehension')
  - 'Scale ID': Scale identifier ('IM' for Importance, 'LV' for Level)
  - 'Data Value': Numerical score indicating importance or level rating

File: "Occupation Data.txt" (Tab-delimited text file or CSV)
Primary Columns Expected:
  - 'O*NET-SOC Code': Unique occupational identifier
  - 'Title': Human-readable occupation title
  - 'Description': Detailed description of the occupation
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd


class ONETLoaderError(Exception):
    """Base exception for O*NET dataset loading failures."""
    pass


class ONETFileNotFoundError(ONETLoaderError, FileNotFoundError):
    """Raised when the expected O*NET file does not exist on disk."""
    pass


class ONETInvalidFormatError(ONETLoaderError, ValueError):
    """Raised when the O*NET file does not contain expected columns."""
    pass


# Documented expected columns in official O*NET Skills export
EXPECTED_ONET_SKILLS_COLUMNS = [
    "O*NET-SOC Code",
    "Element ID",
    "Element Name",
    "Scale ID",
    "Data Value",
]

# Documented expected columns in official O*NET Occupation Data export
EXPECTED_ONET_OCCUPATION_COLUMNS = [
    "O*NET-SOC Code",
    "Title",
]


def load_onet_skills(
    path: Union[str, Path],
    tidy: bool = False,
    scale_id: Optional[str] = "IM",
    occupations_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load and perform validation on O*NET Skills dataset.

    Maintains backwards compatibility with M1 test signature:
    - When `tidy=False`: returns dataframe with raw expected columns:
      ['O*NET-SOC Code', 'Element ID', 'Element Name', 'Scale ID', 'Data Value'].
    - When `tidy=True`: filters to `scale_id` ('IM' for Importance rating by default,
      dropping 'LV' level rows as documented in M2 spec) and returns tidy DataFrame:
      ['onet_soc_code', 'occupation_title', 'skill_name', 'importance_score'].

    Parameters
    ----------
    path : Union[str, Path]
        Path to the raw O*NET Skills file (typically Skills.txt, tab-delimited).
    tidy : bool, default False
        Whether to return a tidy DataFrame standardized for downstream modeling.
    scale_id : Optional[str], default 'IM'
        Scale ID filter when tidy=True. 'IM' corresponds to skill Importance score.
        'LV' (Level/complexity) rows are excluded from importance ranking by default
        because counterfactual gap models prioritize necessity/importance ranking.
    occupations_path : Optional[Union[str, Path]], default None
        Optional path to Occupation Data.txt to merge occupation titles. If not
        provided and tidy=True, attempts to infer 'Occupation Data.txt' in same folder,
        or uses existing 'Title' column if present.

    Returns
    -------
    pd.DataFrame
        Cleaned or tidy DataFrame containing verified O*NET skills schema.

    Raises
    ------
    ONETFileNotFoundError
        If the file at path does not exist.
    ONETInvalidFormatError
        If required columns are missing or the file is improperly formatted.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ONETFileNotFoundError(
            f"O*NET Skills file not found at: '{file_path.resolve()}'. "
            "Please ensure raw O*NET files are downloaded and placed into data/raw/."
        )

    # Detect separator based on extension
    sep = "\t" if file_path.suffix.lower() in [".txt", ".tsv"] else ","
    try:
        df = pd.read_csv(file_path, sep=sep, dtype=str)
    except Exception as e:
        raise ONETInvalidFormatError(f"Failed to parse O*NET Skills file: {e}") from e

    # Clean whitespace in column names if any
    df.columns = [c.strip() for c in df.columns]

    missing_cols = [col for col in EXPECTED_ONET_SKILLS_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ONETInvalidFormatError(
            f"O*NET Skills file missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    if not tidy:
        return df[EXPECTED_ONET_SKILLS_COLUMNS].copy()

    # Tidy transformation:
    # 1. Filter by Scale ID (default 'IM')
    if scale_id is not None and "Scale ID" in df.columns:
        df = df[df["Scale ID"] == scale_id].copy()

    # 2. Resolve occupation titles
    if "Title" not in df.columns:
        # Check if occupations_path provided or sibling file exists
        occ_target = None
        if occupations_path is not None and Path(occupations_path).is_file():
            occ_target = Path(occupations_path)
        else:
            sibling_occ = file_path.parent / "Occupation Data.txt"
            if sibling_occ.is_file():
                occ_target = sibling_occ

        if occ_target is not None:
            occ_df = load_onet_occupations(occ_target)
            df = df.merge(
                occ_df[["onet_soc_code", "occupation_title"]],
                left_on="O*NET-SOC Code",
                right_on="onet_soc_code",
                how="left",
            )
            df["Title"] = df["occupation_title"].fillna("Unknown Occupation")
        else:
            df["Title"] = "Unknown Occupation"

    # 3. Cast Data Value to numeric
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce").fillna(0.0)

    # 4. Standardize output columns
    tidy_df = pd.DataFrame({
        "onet_soc_code": df["O*NET-SOC Code"].astype(str).str.strip(),
        "occupation_title": df["Title"].astype(str).str.strip(),
        "skill_name": df["Element Name"].astype(str).str.strip(),
        "importance_score": df["Data Value"].astype(float),
    })

    return tidy_df.drop_duplicates()


def load_onet_occupations(path: Union[str, Path]) -> pd.DataFrame:
    """Load and validate O*NET Occupation Data export.

    Parameters
    ----------
    path : Union[str, Path]
        Path to Occupation Data.txt or Occupation Data.xlsx.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['onet_soc_code', 'occupation_title', 'description'].

    Raises
    ------
    ONETFileNotFoundError
        If the occupation file does not exist.
    ONETInvalidFormatError
        If required columns ('O*NET-SOC Code', 'Title') are missing.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ONETFileNotFoundError(
            f"O*NET Occupation file not found at: '{file_path.resolve()}'. "
            "Please ensure raw O*NET files are downloaded and placed into data/raw/."
        )

    sep = "\t" if file_path.suffix.lower() in [".txt", ".tsv"] else ","
    try:
        df = pd.read_csv(file_path, sep=sep, dtype=str)
    except Exception as e:
        raise ONETInvalidFormatError(f"Failed to parse O*NET Occupation file: {e}") from e

    df.columns = [c.strip() for c in df.columns]

    missing_cols = [col for col in EXPECTED_ONET_OCCUPATION_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ONETInvalidFormatError(
            f"O*NET Occupation file missing expected columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    desc_col = "Description" if "Description" in df.columns else None
    
    result = pd.DataFrame({
        "onet_soc_code": df["O*NET-SOC Code"].astype(str).str.strip(),
        "occupation_title": df["Title"].astype(str).str.strip(),
    })
    if desc_col:
        result["description"] = df[desc_col].fillna("").astype(str).str.strip()

    return result.drop_duplicates()
