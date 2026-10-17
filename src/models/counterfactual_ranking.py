"""Counterfactual Skill-Gap Ranking Module (Milestone M5).

Simulates hypothetical additions of individual missing skills to a candidate's profile,
evaluating the resulting score delta to prioritize the highest-impact skill interventions.
"""

from typing import Dict, List, Optional, Set, Union
import networkx as nx
import pandas as pd

from src.models.readiness_score import (
    compute_readiness,
    MANDATORY_DISCLAIMER,
    MODEL_WEIGHT_ALPHA,
)


def generate_explanation_text(
    skill_name: str,
    delta: float,
    comp_weight: float,
    current_credit: float,
    matched_via: Optional[str] = None,
    similarity: float = 0.0,
) -> str:
    """Generate deterministic, data-traceable explanation text without external LLMs.

    Parameters
    ----------
    skill_name : str
        The skill being evaluated.
    delta : float
        Predicted readiness score increase.
    comp_weight : float
        Composite requirement weight of the skill in this occupation.
    current_credit : float
        Candidate's current credit for this skill (0.0 or partial).
    matched_via : Optional[str]
        Adjacent skill in profile providing partial credit, if any.
    similarity : float
        Cosine similarity to the adjacent skill.

    Returns
    -------
    str
        Structured explanation string.
    """
    pct_gain = delta * 100.0
    if current_credit > 0.0 and matched_via:
        return (
            f"High-priority capability with requirement weight {comp_weight:.3f}. "
            f"You currently hold adjacent skill '{matched_via}' (similarity: {similarity:.2f}), "
            f"yielding partial credit ({current_credit*100:.0f}%). Full mastery yields a predicted "
            f"+{pct_gain:.2f}% readiness gain."
        )
    elif comp_weight >= 0.75:
        return (
            f"Core critical requirement with high weight {comp_weight:.3f}. "
            f"No related skill was found in your current inventory; acquiring it yields "
            f"a top-tier predicted readiness boost of +{pct_gain:.2f}%."
        )
    elif comp_weight >= 0.50:
        return (
            f"Substantial operational requirement (weight: {comp_weight:.3f}). "
            f"Acquiring this competency directly closes a primary qualification gap (+{pct_gain:.2f}% readiness)."
        )
    else:
        return (
            f"Secondary/discretionary competency (weight: {comp_weight:.3f}). "
            f"Provides an incremental readiness improvement of +{pct_gain:.2f}%."
        )


def rank_skill_gaps(
    candidate_skills: Set[str],
    occupation_id: str,
    model_artifact: Optional[Dict] = None,
    graph: Optional[nx.Graph] = None,
    top_n: int = 10,
    alpha: float = MODEL_WEIGHT_ALPHA,
) -> pd.DataFrame:
    """Simulate single-skill interventions and rank missing skills by predicted readiness delta.

    Parameters
    ----------
    candidate_skills : Set[str]
        Set of skill names currently possessed by the candidate.
    occupation_id : str
        Target occupation node ID or original ID.
    model_artifact : Optional[Dict], default None
        Loaded M4 baseline model artifact.
    graph : Optional[nx.Graph], default None
        Heterogeneous skill and occupation graph.
    top_n : int, default 10
        Number of top skill gaps to return.
    alpha : float, default 0.50
        Balance parameter for requirement weight.

    Returns
    -------
    pd.DataFrame
        Ranked skills DataFrame with columns:
        ['skill_name', 'baseline_score', 'hypothetical_score', 'predicted_delta',
         'requirement_weight', 'explanation_text']
        Strictly excludes skills the candidate already possesses directly.
    """
    candidate_skills_norm = {s.lower().strip() for s in candidate_skills if isinstance(s, str)}

    # 1. Compute baseline readiness
    baseline_score, breakdown = compute_readiness(
        candidate_skills=candidate_skills,
        occupation_id=occupation_id,
        model_artifact=model_artifact,
        graph=graph,
        alpha=alpha,
    )

    gap_candidates = []

    # 2. Iterate through required skills
    for s_name, details in breakdown.items():
        # Exclude skills already possessed directly (credit == 1.0)
        if details["is_present"] or s_name.lower().strip() in candidate_skills_norm:
            continue

        # Simulate hypothetical addition of s_name
        hypothetical_skills = set(candidate_skills).union({s_name})
        hypo_score, _ = compute_readiness(
            candidate_skills=hypothetical_skills,
            occupation_id=occupation_id,
            model_artifact=model_artifact,
            graph=graph,
            alpha=alpha,
        )

        delta = max(hypo_score - baseline_score, 0.0)
        comp_w = details["composite_weight"]
        curr_credit = details["credit"]
        matched_via = details["matched_via"]
        sim = details["similarity"]

        explanation = generate_explanation_text(
            skill_name=s_name,
            delta=delta,
            comp_weight=comp_w,
            current_credit=curr_credit,
            matched_via=matched_via,
            similarity=sim,
        )

        gap_candidates.append({
            "skill_name": s_name,
            "baseline_score": baseline_score,
            "hypothetical_score": hypo_score,
            "predicted_delta": round(float(delta), 4),
            "requirement_weight": round(float(comp_w), 4),
            "explanation_text": explanation,
        })

    if not gap_candidates:
        return pd.DataFrame(columns=[
            "skill_name", "baseline_score", "hypothetical_score",
            "predicted_delta", "requirement_weight", "explanation_text"
        ])

    df_ranked = pd.DataFrame(gap_candidates)
    df_ranked.sort_values(by=["predicted_delta", "requirement_weight"], ascending=[False, False], inplace=True)
    df_ranked.reset_index(drop=True, inplace=True)

    return df_ranked.head(top_n)
