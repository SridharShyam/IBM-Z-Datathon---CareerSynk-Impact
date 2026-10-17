"""Skill normalization and cross-taxonomy reconciliation utilities.

This module provides tools to normalize skill strings and perform cross-taxonomy
reconciliation between O*NET and ESCO skill sets using lexical string matching.

Note:
The lexical matching function `reconcile_onet_esco_lexical` is maintained and tested
for ablation and comparison purposes. For production 1-to-many semantic skill mapping,
use `src.preprocessing.semantic_reconciliation.reconcile_onet_esco_semantic`.
"""

from pathlib import Path
import re
from typing import Optional, Set, Tuple, Union
import pandas as pd
from rapidfuzz import fuzz, process

# Named constant for fuzzy match threshold (documented rationale:
# Scores >= 82.0 on token-sort-ratio provide high confidence matches
# across verb-noun order variations, but can create false-friend compound collisions.
# Maintained for ablation comparison against semantic reconciliation).
FUZZY_MATCH_THRESHOLD: float = 82.0


def normalize_skill_name(name: str) -> str:
    """Normalize a raw skill name into a standardized string representation.

    Transformations applied:
      1. Converts to lowercase.
      2. Strips leading and trailing whitespace.
      3. Normalizes special punctuation/delimiters (dashes, slashes, underscores) to spaces.
      4. Strips out extraneous non-alphanumeric punctuation except essential characters.
      5. Collapses multiple consecutive whitespaces into a single space.

    Parameters
    ----------
    name : str
        The raw skill string to normalize.

    Returns
    -------
    str
        Cleaned, normalized skill identifier string.
    """
    if not isinstance(name, str):
        return ""

    # Convert to lowercase and strip
    text = name.lower().strip()

    # Replace dashes, underscores, and slashes with spaces to prevent merging words
    text = re.sub(r"[\-_/\\|]", " ", text)

    # Remove unwanted punctuation (preserve alphanumerics and single spaces)
    text = re.sub(r"[^\w\s]", "", text)

    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


def reconcile_onet_esco_lexical(
    onet_df: pd.DataFrame,
    esco_df: pd.DataFrame,
    threshold: float = FUZZY_MATCH_THRESHOLD,
    report_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Reconcile O*NET and ESCO skill names using token-sort lexical matching (M2 Baseline).

    Maintained for ablation and side-by-side comparison with semantic reconciliation.

    Parameters
    ----------
    onet_df : pd.DataFrame
        DataFrame of loaded O*NET skills. Expected to contain 'skill_name' (or 'Element Name').
    esco_df : pd.DataFrame
        DataFrame of loaded ESCO skills. Expected to contain 'skill_name' (or 'preferredLabel').
    threshold : float, default FUZZY_MATCH_THRESHOLD (82.0)
        Minimum RapidFuzz token-sort-ratio similarity score (0-100) to accept a fuzzy match.
    report_path : Optional[Union[str, Path]], default None
        Optional path to write the markdown reconciliation audit report.

    Returns
    -------
    pd.DataFrame
        A `skill_crosswalk` DataFrame with columns:
        ['onet_skill_name', 'esco_skill_name', 'match_type', 'match_score'].
    """
    # 1. Resolve skill column in onet_df
    onet_col = "skill_name" if "skill_name" in onet_df.columns else "Element Name"
    if onet_col not in onet_df.columns:
        raise ValueError(f"onet_df missing skill column. Found: {list(onet_df.columns)}")

    # 2. Resolve skill column in esco_df
    esco_col = "skill_name" if "skill_name" in esco_df.columns else "preferredLabel"
    if esco_col not in esco_df.columns:
        raise ValueError(f"esco_df missing skill column. Found: {list(esco_df.columns)}")

    # Extract distinct non-empty raw skills
    onet_raw_unique = [str(s).strip() for s in onet_df[onet_col].dropna().unique() if str(s).strip()]
    esco_raw_unique = [str(s).strip() for s in esco_df[esco_col].dropna().unique() if str(s).strip()]

    # Normalize skill names while mapping back to original canonical string
    onet_norm_to_raw = {normalize_skill_name(s): s for s in onet_raw_unique}
    esco_norm_to_raw = {normalize_skill_name(s): s for s in esco_raw_unique}

    esco_norm_keys = list(esco_norm_to_raw.keys())
    matched_esco_norm_keys: Set[str] = set()

    crosswalk_records = []
    exact_count = 0
    fuzzy_count = 0
    unmatched_onet_count = 0

    # Match each O*NET skill
    for onet_norm, onet_orig in onet_norm_to_raw.items():
        # Exact matching check
        if onet_norm in esco_norm_to_raw:
            matched_esco = esco_norm_to_raw[onet_norm]
            matched_esco_norm_keys.add(onet_norm)
            crosswalk_records.append({
                "onet_skill_name": onet_orig,
                "esco_skill_name": matched_esco,
                "match_type": "exact",
                "match_score": 100.0,
            })
            exact_count += 1
            continue

        # Fuzzy matching check using token_sort_ratio
        best_match = process.extractOne(
            onet_norm,
            esco_norm_keys,
            scorer=fuzz.token_sort_ratio,
        )

        if best_match and best_match[1] >= threshold:
            best_esco_norm, score, _ = best_match
            matched_esco = esco_norm_to_raw[best_esco_norm]
            matched_esco_norm_keys.add(best_esco_norm)
            crosswalk_records.append({
                "onet_skill_name": onet_orig,
                "esco_skill_name": matched_esco,
                "match_type": "fuzzy",
                "match_score": float(score),
            })
            fuzzy_count += 1
        else:
            crosswalk_records.append({
                "onet_skill_name": onet_orig,
                "esco_skill_name": None,
                "match_type": "unmatched",
                "match_score": 0.0,
            })
            unmatched_onet_count += 1

    # Add unmatched ESCO skills into the crosswalk so no skill is dropped
    unmatched_esco_count = 0
    for esco_norm, esco_orig in esco_norm_to_raw.items():
        if esco_norm not in matched_esco_norm_keys:
            crosswalk_records.append({
                "onet_skill_name": None,
                "esco_skill_name": esco_orig,
                "match_type": "unmatched",
                "match_score": 0.0,
            })
            unmatched_esco_count += 1

    crosswalk_df = pd.DataFrame(crosswalk_records)

    # 3. Log audit report if requested
    if report_path:
        out_path = Path(report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_md = f"""# O*NET - ESCO Skill Reconciliation Report (Lexical M2)
*Generated by CareerSynk Impact*

## 1. Executive Summary
- **Total Unique O*NET Skills**: {len(onet_raw_unique)}
- **Total Unique ESCO Skills**: {len(esco_raw_unique)}
- **Exact Matches Found**: {exact_count}
- **Fuzzy Matches Found (>= {threshold})**: {fuzzy_count}
- **Unmatched O*NET Skills**: {unmatched_onet_count}
- **Unmatched ESCO Skills**: {unmatched_esco_count}
- **Total Crosswalk Entries**: {len(crosswalk_df)}

## 2. Reconciled O*NET Matches
| O*NET Skill | Matched ESCO Skill | Match Type | Match Score |
| :--- | :--- | :--- | :--- |
"""
        matched_sample = crosswalk_df[crosswalk_df["match_type"].isin(["exact", "fuzzy"])]
        for _, row in matched_sample.iterrows():
            report_md += f"| {row['onet_skill_name']} | {row['esco_skill_name']} | {row['match_type']} | {row['match_score']:.1f} |\n"

        out_path.write_text(report_md, encoding="utf-8")

    return crosswalk_df


# Backwards compatibility alias for M1/M2 tests
reconcile_onet_esco = reconcile_onet_esco_lexical
