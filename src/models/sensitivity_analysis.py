"""Readiness Sensitivity Analysis across Alpha Parameters.

Evaluates how stable counterfactual skill-gap rankings are under varying alpha
weightings in the composite requirement weight formula:
    W_s = alpha * w(O, s) + (1 - alpha) * r_hat(O, s)
for alpha in [0.2, 0.5, 0.8].
"""

import os
import pickle
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

from src.models.counterfactual_ranking import rank_skill_gaps
from src.models.demo_profiles import SYNTHETIC_PROFILES, TARGET_OCCUPATIONS

ALPHA_VALUES = [0.2, 0.5, 0.8]


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity index between two skill sets: |A & B| / |A | B|."""
    if not set_a and not set_b:
        return 1.0
    union = set_a.union(set_b)
    if not union:
        return 1.0
    return len(set_a.intersection(set_b)) / len(union)


def run_sensitivity_analysis(
    profiles: Dict[str, Dict],
    target_occupations: List[Dict[str, str]],
    model_artifact: Dict,
    graph,
    alphas: List[float] = ALPHA_VALUES,
    top_k: int = 5,
) -> Dict:
    """Run sensitivity analysis across alpha values for profiles and target occupations.

    Parameters
    ----------
    profiles : Dict[str, Dict]
        Candidate profile dictionary.
    target_occupations : List[Dict[str, str]]
        List of target occupation dicts with 'id' and 'title'.
    model_artifact : Dict
        Loaded Stage A model artifact.
    graph : nx.Graph
        Heterogeneous skill graph.
    alphas : List[float]
        List of alpha weights to test (default [0.2, 0.5, 0.8]).
    top_k : int
        Top-k skill gaps to evaluate (default 5).

    Returns
    -------
    Dict
        Structured results containing rankings, pairwise Jaccard similarities,
        and summary metrics.
    """
    results = {}

    for occ in target_occupations:
        occ_id = occ["id"]
        occ_title = occ["title"]
        results[occ_id] = {
            "title": occ_title,
            "profiles": {},
        }

        for prof_id, prof_data in profiles.items():
            candidate_skills = set(prof_data["skills"])
            alpha_rankings: Dict[float, List[str]] = {}

            for alpha in alphas:
                df_ranked = rank_skill_gaps(
                    candidate_skills=candidate_skills,
                    occupation_id=occ_id,
                    model_artifact=model_artifact,
                    graph=graph,
                    top_n=top_k,
                    alpha=alpha,
                )
                top_skills = df_ranked["skill_name"].tolist() if not df_ranked.empty else []
                alpha_rankings[alpha] = top_skills

            # Pairwise Jaccard metrics across the extreme and midpoint alphas
            # Primary stability metric: Jaccard(alpha=0.2, alpha=0.8) and Jaccard(alpha=0.5, alpha=0.8)
            set_02 = set(alpha_rankings.get(0.2, []))
            set_05 = set(alpha_rankings.get(0.5, []))
            set_08 = set(alpha_rankings.get(0.8, []))

            jaccard_02_08 = compute_jaccard_similarity(set_02, set_08)
            jaccard_02_05 = compute_jaccard_similarity(set_02, set_05)
            jaccard_05_08 = compute_jaccard_similarity(set_05, set_08)

            results[occ_id]["profiles"][prof_id] = {
                "profile_name": prof_data.get("name", prof_id),
                "rankings": alpha_rankings,
                "jaccard_02_08": round(jaccard_02_08, 4),
                "jaccard_02_05": round(jaccard_02_05, 4),
                "jaccard_05_08": round(jaccard_05_08, 4),
            }

    return results


def generate_markdown_report(analysis_results: Dict, output_path: str) -> str:
    """Generate comprehensive markdown sensitivity report and write to file."""
    lines = []
    lines.append("# Sensitivity Analysis: Counterfactual Skill-Gap Ranking Stability")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(
        "This report assesses the stability of top-5 counterfactual skill-gap recommendations "
        "when shifting the composite requirement weight parameter $\\alpha \\in [0.2, 0.5, 0.8]$ "
        "in the formula $W_s = \\alpha \\cdot w(O, s) + (1 - \\alpha) \\cdot \\hat{r}(O, s)$."
    )
    lines.append(
        "- **$\\alpha = 0.20$**: Strongly weights ML occupational generalization $\\hat{r}(O, s)$."
    )
    lines.append(
        "- **$\\alpha = 0.50$ (Shipped Default)**: Balanced compromise between empirical requirement strength and ML relevance."
    )
    lines.append(
        "- **$\\alpha = 0.80$**: Strongly weights observed structural requirement weight $w(O, s)$."
    )
    lines.append("")
    lines.append("### Primary Metric: Top-5 Jaccard Set Overlap")
    lines.append(
        "**Metric Choice Justification**: In workforce upskilling systems, recommendations are surfaced "
        "as a cohort of high-priority learning interventions (e.g. top-5 missing skills). Whether a skill appears "
        "within the actionable top-5 intervention set is significantly more impactful to the candidate than a minor "
        "rank permutation between position #2 and #3. Therefore, **Top-5 Jaccard Set Overlap** "
        "($J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$) between $\\alpha=0.2$ and $\\alpha=0.8$ is chosen as the "
        "primary stability metric, bounded in $[0.0, 1.0]$."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    all_jaccard_02_08 = []

    for occ_id, occ_data in analysis_results.items():
        lines.append(f"## Target Occupation: {occ_data['title']} (`{occ_id}`)")
        lines.append("")
        lines.append("| Candidate Profile | $\\alpha=0.2$ Top-5 Skills | $\\alpha=0.5$ (Default) Top-5 Skills | $\\alpha=0.8$ Top-5 Skills | $J(\\alpha_{0.2}, \\alpha_{0.8})$ | $J(\\alpha_{0.5}, \\alpha_{0.8})$ |")
        lines.append("| :--- | :--- | :--- | :--- | :---: | :---: |")

        for prof_id, pdata in occ_data["profiles"].items():
            p_name = pdata["profile_name"]
            r02 = "<br>".join([f"{i+1}. {s}" for i, s in enumerate(pdata["rankings"].get(0.2, []))])
            r05 = "<br>".join([f"{i+1}. {s}" for i, s in enumerate(pdata["rankings"].get(0.5, []))])
            r08 = "<br>".join([f"{i+1}. {s}" for i, s in enumerate(pdata["rankings"].get(0.8, []))])
            j0208 = pdata["jaccard_02_08"]
            j0508 = pdata["jaccard_05_08"]
            all_jaccard_02_08.append(j0208)

            lines.append(f"| **{p_name}** | {r02} | {r05} | {r08} | **{j0208:.2f}** | **{j0508:.2f}** |")

        lines.append("")

    mean_jaccard = float(np.mean(all_jaccard_02_08)) if all_jaccard_02_08 else 0.0

    lines.append("## Plain-English Conclusion & Shipped Default Decision")
    lines.append(
        f"Across all synthetic candidate profiles and target occupations, the mean Top-5 Jaccard stability "
        f"between extreme weighting regimes ($\\alpha=0.2$ vs $\\alpha=0.8$) is **{mean_jaccard:.2f}**."
    )
    if mean_jaccard >= 0.70:
        lines.append(
            "The top-5 counterfactual recommendations exhibit **high stability** across the entire $\\alpha$ "
            "spectrum. While individual readiness deltas adjust smoothly in magnitude, the core set of high-leverage "
            "skill interventions remains robust to the relative weight assigned to observed occupational graphs versus "
            "ML model generalization."
        )
    else:
        lines.append(
            "The top-5 recommendations exhibit **moderate sensitivity** to the choice of $\\alpha$, with some skill "
            "rank re-orderings occurring when moving from purely observed requirement weighting ($\\alpha=0.8$) "
            "to model-heavy generalization ($\\alpha=0.2$)."
        )

    lines.append(
        "\n**Shipped Default Recommendation**: We **retain $\\alpha = 0.50$** as the shipped default. "
        "At $\\alpha = 0.50$, the system balances verified ground-truth requirements from the empirical occupational "
        "profile with the Stage A model's ability to smooth noise and penalize overly generic skills, without "
        "skewing recommendations exclusively toward either source."
    )
    lines.append("")
    lines.append("---")
    lines.append("### Mandatory Non-Causal Framing Disclaimer")
    lines.append(
        "> **Non-Causal Disclaimer**: Readiness scores and counterfactual deltas are statistical associations "
        "derived from cross-sectional occupational data. They indicate correlation with occupational skill requirements "
        "and transferable aptitude, not a guarantee of hiring, career advancement, or causal wage uplift."
    )
    lines.append("")

    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    graph_path = os.path.join(base_dir, "data", "processed", "skill_graph.pkl")
    model_path = os.path.join(base_dir, "data", "processed", "baseline_relevance_model.pkl")
    report_path = os.path.join(base_dir, "data", "processed", "sensitivity_report.md")

    print(f"Loading graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    print(f"Loading model artifact from {model_path}...")
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    print("Running sensitivity analysis across alpha = [0.2, 0.5, 0.8]...")
    results = run_sensitivity_analysis(
        profiles=SYNTHETIC_PROFILES,
        target_occupations=TARGET_OCCUPATIONS,
        model_artifact=model_artifact,
        graph=graph,
        alphas=ALPHA_VALUES,
        top_k=5,
    )

    print(f"Generating sensitivity report at {report_path}...")
    generate_markdown_report(results, report_path)
    print("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
