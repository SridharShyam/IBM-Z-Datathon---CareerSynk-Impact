"""Candidate Career Readiness Scoring Module (Milestone M5).

Defines the mathematical formulation and computation for evaluating a candidate's
readiness score relative to a target occupation's skill requirements.

Mathematical Formulation:
Given:
  - Target occupation O with a set of required skills S_O.
  - Candidate's verified skill inventory C (set of normalized skill strings).
  - Graph edge requirement weight w(O, s) for each skill s in S_O.
    * For O*NET: w(O, s) is the analyst-verified importance rating normalized to [0.0, 1.0].
    * For ESCO: w(O, s) is mapped explicitly as essential=1.0, optional=0.5.
  - Model predicted relevance score r_hat(O, s) in [0.0, 1.0] from the Stage A ML model.
    The composite requirement importance for skill s is:
      W_s = alpha * w(O, s) + (1 - alpha) * r_hat(O, s)
    (default alpha = 0.5 balances observed requirement strength with learned model generalization).

Partial Credit via Semantic Skill Graph:
If candidate does not possess skill s directly (s not in C):
  - Let N_related(s) be the set of skills connected to s in the heterogeneous skill graph
    via 'possibly_related' or 'related_to' edges, where edge weight sim(s, s') >= PARTIAL_CREDIT_MIN_SIM (default 0.70).
  - If any s' in N_related(s) is in candidate skills C:
      credit(s) = max_{s' in (C cap N_related(s))} [ sim(s, s') * PARTIAL_CREDIT_DISCOUNT ]
      (default PARTIAL_CREDIT_DISCOUNT = 0.50 reflects that an adjacent competency
      provides transferable foundational aptitude but does not fully replace direct competence).
  - If s in C:
      credit(s) = 1.0
  - Otherwise:
      credit(s) = 0.0

Readiness Score Formula:
  Readiness(C, O) = [ sum_{s in S_O} (W_s * credit(s)) ] / [ sum_{s in S_O} W_s ]

Properties:
  - Bounded in [0.0, 1.0].
  - Monotonic: Acquiring a new skill strictly increases (or maintains) readiness.
  - Sensitive to priority: Gaps in high-importance skills penalize readiness far more
    than gaps in low-importance or optional skills.

Mandatory Limitation Disclaimer:
These scores represent model-based statistical associations and heuristic readiness
estimates, NOT causal claims about employability, hiring probability, or compensation.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np

# Modeling constants
PARTIAL_CREDIT_MIN_SIM: float = 0.70
# Minimum semantic similarity to grant partial credit for an adjacent skill

PARTIAL_CREDIT_DISCOUNT: float = 0.50
# Discount factor applied to adjacent skills (a closely related skill awards at most 50% credit)

MODEL_WEIGHT_ALPHA: float = 0.50
# Balance between ground-truth observed weight w(O, s) and model-predicted relevance r_hat(O, s)

MANDATORY_DISCLAIMER: str = (
    "DISCLAIMER: These are model-based predicted associations and score deltas, "
    "not causal claims about employability, hiring probability, or salary outcomes. "
    "The underlying model's accuracy is bounded by the M4 evaluation metrics (MAE 0.068, "
    "mean Spearman 0.88), and occupation-specific accuracy may vary."
)


def compute_skill_credit(
    target_skill_node: str,
    target_skill_label: str,
    candidate_skills_norm: Set[str],
    graph: Optional[nx.Graph] = None,
    min_sim: float = PARTIAL_CREDIT_MIN_SIM,
    discount: float = PARTIAL_CREDIT_DISCOUNT,
) -> Tuple[float, Optional[str], float]:
    """Calculate the candidate's credit for a specific target skill.

    Returns
    -------
    Tuple[float, Optional[str], float]
        credit (0.0 to 1.0), matched_skill_label, similarity_score
    """
    # 1. Direct match check
    if target_skill_label.lower().strip() in candidate_skills_norm:
        return 1.0, target_skill_label, 1.0

    # 2. Check partial credit via graph neighbors
    if graph is not None and graph.has_node(target_skill_node):
        best_credit = 0.0
        best_neighbor = None
        best_sim = 0.0

        for neighbor in graph.neighbors(target_skill_node):
            edge_data = graph.get_edge_data(target_skill_node, neighbor)
            edge_type = edge_data.get("edge_type", "")
            if edge_type in ["possibly_related", "related_to"]:
                sim = float(edge_data.get("weight", 0.0))
                if sim >= min_sim:
                    n_label = str(graph.nodes[neighbor].get("label", "")).lower().strip()
                    if n_label in candidate_skills_norm:
                        cand_credit = sim * discount
                        if cand_credit > best_credit:
                            best_credit = cand_credit
                            best_neighbor = graph.nodes[neighbor].get("label", neighbor)
                            best_sim = sim

        if best_credit > 0.0:
            return best_credit, best_neighbor, best_sim

    return 0.0, None, 0.0


def compute_readiness(
    candidate_skills: Set[str],
    occupation_id: str,
    model_artifact: Optional[Dict] = None,
    graph: Optional[nx.Graph] = None,
    alpha: float = MODEL_WEIGHT_ALPHA,
) -> Tuple[float, Dict[str, Dict]]:
    """Compute normalized candidate readiness score for a target occupation.

    Parameters
    ----------
    candidate_skills : Set[str]
        Set of skill names currently possessed by the candidate.
    occupation_id : str
        Target occupation node ID in graph (e.g. 'occ:onet:15-1252.00') or original ID ('15-1252.00').
    model_artifact : Optional[Dict], default None
        Loaded M4 baseline model artifact containing trained model, TF-IDF, and skill embeddings.
    graph : Optional[nx.Graph], default None
        Heterogeneous skill and occupation graph from M4.
    alpha : float, default 0.50
        Weight between observed graph edge weight and model predicted relevance.

    Returns
    -------
    Tuple[float, Dict[str, Dict]]
        Overall readiness score in [0.0, 1.0], and detailed per-skill breakdown dictionary.
    """
    candidate_skills_norm = {s.lower().strip() for s in candidate_skills if isinstance(s, str)}

    # Resolve occupation node in graph
    target_node = occupation_id
    if graph is not None:
        if not graph.has_node(target_node):
            # Check prefixes
            candidates = [n for n in graph.nodes if n.endswith(f":{occupation_id}") or n == occupation_id]
            if candidates:
                target_node = candidates[0]
            else:
                raise ValueError(f"Occupation ID '{occupation_id}' not found in graph.")

    # Collect required skills from graph
    required_skills_data = []
    if graph is not None and graph.has_node(target_node):
        for neighbor in graph.neighbors(target_node):
            edge_data = graph.get_edge_data(target_node, neighbor)
            if edge_data.get("edge_type") == "requires":
                s_label = str(graph.nodes[neighbor].get("label", neighbor))
                obs_weight = float(edge_data.get("weight", 0.5))
                required_skills_data.append({
                    "skill_node": neighbor,
                    "skill_name": s_label,
                    "observed_weight": obs_weight,
                })
    else:
        raise ValueError("A populated graph is required to determine occupation skill requirements.")

    if not required_skills_data:
        return 0.0, {}

    # Extract model predicted relevance if model_artifact is available
    model = model_artifact.get("model") if model_artifact else None
    tfidf = model_artifact.get("tfidf") if model_artifact else None
    skill_emb_map = model_artifact.get("skill_emb_map") if model_artifact else None

    # Occupation text representation for model inference
    occ_label = graph.nodes[target_node].get("label", "") if graph else ""
    occ_text = f"{occ_label}: {occ_label}"

    total_weight = 0.0
    accumulated_credit = 0.0
    per_skill_details = {}

    for req in required_skills_data:
        s_node = req["skill_node"]
        s_name = req["skill_name"]
        w_obs = req["observed_weight"]

        # Predict relevance via model if available, else fallback to observed weight
        if model is not None and skill_emb_map is not None and s_name in skill_emb_map:
            try:
                s_emb = skill_emb_map[s_name].reshape(1, -1)
                occ_feat = tfidf.transform([occ_text]).toarray() if tfidf else np.zeros((1, 64))
                # Semantic interaction
                interact = np.sum(s_emb[:, :64] * occ_feat, axis=1, keepdims=True)
                # Feature vector shape (1, 449)
                feat = np.hstack([s_emb, occ_feat, interact])
                r_pred = float(model.predict(feat)[0])
                r_pred = min(max(r_pred, 0.0), 1.0)
            except Exception:
                r_pred = w_obs
        else:
            r_pred = w_obs

        # Composite requirement weight W_s
        W_s = alpha * w_obs + (1.0 - alpha) * r_pred

        # Compute credit
        credit, matched_neighbor, sim = compute_skill_credit(
            target_skill_node=s_node,
            target_skill_label=s_name,
            candidate_skills_norm=candidate_skills_norm,
            graph=graph,
        )

        total_weight += W_s
        accumulated_credit += W_s * credit

        per_skill_details[s_name] = {
            "skill_node": s_node,
            "observed_weight": w_obs,
            "predicted_relevance": r_pred,
            "composite_weight": W_s,
            "credit": credit,
            "matched_via": matched_neighbor,
            "similarity": sim,
            "is_present": (credit == 1.0),
        }

    overall_readiness = (accumulated_credit / total_weight) if total_weight > 0 else 0.0
    return round(float(overall_readiness), 4), per_skill_details
