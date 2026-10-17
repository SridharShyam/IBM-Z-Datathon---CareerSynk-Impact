"""Skill and Occupation Graph Builder (Milestone M4).

Constructs an occupation-skill heterogeneous graph using NetworkX from:
  1. data/processed/occupation_skills.parquet
  2. data/processed/skill_crosswalk.parquet

Node Types:
  - 'occupation': O*NET and ESCO occupations (source-tagged, distinct namespaces).
  - 'skill': Unified or linked skills from O*NET and ESCO.
    If crosswalk embedding_score >= MERGE_CONFIDENCE_THRESHOLD, O*NET and ESCO skill
    concepts are unified into a canonical skill node.
    Otherwise, they remain separate skill nodes connected by a 'possibly_related' edge.

Edge Types:
  - occupation -> skill ('requires'):
      * O*NET weight: Importance score normalized to [0, 1] (score / 5.0).
      * ESCO weight: Model assumption: essential = 1.0, optional = 0.5.
  - skill -> skill ('related_to' or 'possibly_related'):
      * 'related_to': High-confidence crosswalk edge (score >= MERGE_CONFIDENCE_THRESHOLD).
      * 'possibly_related': Candidate edge (score < MERGE_CONFIDENCE_THRESHOLD).
      * NOTE: These edges represent semantic embedding-based relatedness ONLY, NOT
        validated curriculum, chronological, or causal prerequisite ordering.
"""

from pathlib import Path
import pickle
import sys
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
import pandas as pd

# Named threshold for skill-node unification
# Rationale: Cosine embedding similarity >= 0.80 indicates highly aligned conceptual
# equivalence (e.g. 'Active Listening' <-> 'listen actively' at 0.841, or 'Mathematics' at 1.0).
# Below 0.80, skills represent related sub-competencies or contextual specializations,
# so they are kept as distinct nodes connected by 'possibly_related' edges rather than merged.
MERGE_CONFIDENCE_THRESHOLD: float = 0.80

# ESCO categorical relation to numeric weight mapping assumption:
# Essential relations represent core required job capabilities (1.0).
# Optional relations represent discretionary or domain-specific value-adds (0.5).
ESCO_RELATION_WEIGHTS: Dict[str, float] = {
    "essential": 1.0,
    "optional": 0.5,
}


def build_skill_graph(
    occupation_skills_path: Union[str, Path],
    crosswalk_path: Union[str, Path],
    merge_threshold: float = MERGE_CONFIDENCE_THRESHOLD,
    output_graph_path: Optional[Union[str, Path]] = None,
    sanity_check_path: Optional[Union[str, Path]] = None,
) -> Tuple[nx.Graph, Dict[str, Union[int, Dict[str, int]]]]:
    """Build heterogeneous occupation-skill graph with typed nodes and edges.

    Parameters
    ----------
    occupation_skills_path : Union[str, Path]
        Path to data/processed/occupation_skills.parquet.
    crosswalk_path : Union[str, Path]
        Path to data/processed/skill_crosswalk.parquet.
    merge_threshold : float, default 0.80
        Embedding score threshold for unifying O*NET and ESCO skill nodes.
    output_graph_path : Optional[Union[str, Path]], default None
        File path to serialize the generated graph (pickle or GraphML).
    sanity_check_path : Optional[Union[str, Path]], default None
        File path to write the markdown sanity check report.

    Returns
    -------
    Tuple[nx.Graph, Dict[str, Union[int, Dict[str, int]]]]
        The NetworkX graph and a dictionary of node/edge summary metrics.
    """
    occ_skills_df = pd.read_parquet(occupation_skills_path)
    crosswalk_df = pd.read_parquet(crosswalk_path)

    G = nx.Graph()

    # 1. Process crosswalk to establish skill node unification and skill-skill edges
    # Build mapping for skills unified above merge_threshold
    # onet_skill -> canonical_id, esco_skill -> canonical_id
    canonical_skill_map: Dict[str, str] = {}
    skill_skill_edges: List[Tuple[str, str, Dict]] = []

    for _, row in crosswalk_df.iterrows():
        onet_name = str(row.get("onet_skill_name", "")).strip()
        esco_name = str(row.get("esco_skill_name", "")).strip()
        score = float(row.get("embedding_score", 0.0))
        h_sup = bool(row.get("hierarchy_supported", False))

        if not onet_name or not esco_name or esco_name == "None" or pd.isna(esco_name):
            continue

        onet_node_id = f"skill:onet:{onet_name}"
        esco_node_id = f"skill:esco:{esco_name}"

        if score >= merge_threshold:
            # Unify nodes: canonical ID will represent the unified concept
            unified_id = f"skill:unified:{onet_name}"
            canonical_skill_map[onet_node_id] = unified_id
            canonical_skill_map[esco_node_id] = unified_id

            # Add unified node if not present
            if not G.has_node(unified_id):
                G.add_node(
                    unified_id,
                    node_type="skill",
                    label=onet_name,
                    source="unified",
                    onet_name=onet_name,
                    esco_name=esco_name,
                    merge_score=score,
                    hierarchy_supported=h_sup,
                )
        else:
            # Below threshold: keep separate nodes and link with 'possibly_related' edge
            if not G.has_node(onet_node_id):
                G.add_node(
                    onet_node_id,
                    node_type="skill",
                    label=onet_name,
                    source="onet",
                )
            if not G.has_node(esco_node_id):
                G.add_node(
                    esco_node_id,
                    node_type="skill",
                    label=esco_name,
                    source="esco",
                )

            skill_skill_edges.append((
                canonical_skill_map.get(onet_node_id, onet_node_id),
                canonical_skill_map.get(esco_node_id, esco_node_id),
                {
                    "edge_type": "possibly_related",
                    "weight": score,
                    "hierarchy_supported": h_sup,
                },
            ))

    # Add skill-skill edges
    for u, v, attrs in skill_skill_edges:
        if u != v:
            G.add_edge(u, v, **attrs)

    # 2. Add occupation nodes and occupation -> skill edges
    for _, row in occ_skills_df.iterrows():
        occ_id = str(row["occupation_id"]).strip()
        occ_title = str(row["occupation_title"]).strip()
        skill_name = str(row["skill_name"]).strip()
        source = str(row["source"]).strip().lower()
        rel_weight_raw = row["relation_or_weight"]

        occ_node_id = f"occ:{source}:{occ_id}"
        if not G.has_node(occ_node_id):
            G.add_node(
                occ_node_id,
                node_type="occupation",
                label=occ_title,
                source=source,
                original_id=occ_id,
            )

        # Resolve skill node target
        raw_skill_node_id = f"skill:{source}:{skill_name}"
        skill_target_id = canonical_skill_map.get(raw_skill_node_id, raw_skill_node_id)

        if not G.has_node(skill_target_id):
            G.add_node(
                skill_target_id,
                node_type="skill",
                label=skill_name,
                source=source,
            )

        # Compute numerical edge weight
        if source == "onet":
            # Normalize O*NET Importance score (scale 1.0 to 5.0) to [0, 1]
            try:
                num_score = float(rel_weight_raw)
                weight = min(max(num_score / 5.0, 0.0), 1.0)
            except (ValueError, TypeError):
                weight = 0.5
        else:
            # ESCO relation type mapping assumption
            weight = ESCO_RELATION_WEIGHTS.get(str(rel_weight_raw).lower(), 0.5)

        G.add_edge(
            occ_node_id,
            skill_target_id,
            edge_type="requires",
            weight=round(weight, 4),
            source=source,
        )

    # 3. Calculate metrics and node/edge breakdown
    node_counts = {"occupation": 0, "skill": 0}
    for _, data in G.nodes(data=True):
        nt = data.get("node_type", "unknown")
        node_counts[nt] = node_counts.get(nt, 0) + 1

    edge_counts = {"requires": 0, "possibly_related": 0}
    for _, _, data in G.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_counts[et] = edge_counts.get(et, 0) + 1

    summary_metrics = {
        "merge_confidence_threshold": merge_threshold,
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "nodes_by_type": node_counts,
        "edges_by_type": edge_counts,
    }

    # 4. Serialize graph artifact
    if output_graph_path:
        out_p = Path(output_graph_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".pkl":
            with open(out_p, "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif out_p.suffix.lower() == ".graphml":
            nx.write_graphml(G, out_p)
        else:
            with open(out_p.with_suffix(".pkl"), "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 5. Generate sanity check report
    if sanity_check_path:
        generate_graph_sanity_check(G, Path(sanity_check_path), summary_metrics)

    return G, summary_metrics


def generate_graph_sanity_check(
    G: nx.Graph,
    report_path: Path,
    summary_metrics: Dict,
):
    """Generate markdown sanity check inspecting top required skills for sample occupations."""
    sample_queries = [
        ("occ:onet:11-1011.00", "Chief Executives (O*NET)"),
        ("occ:onet:15-1252.00", "Software Developers (O*NET)"),
        ("occ:esco:http://data.europa.eu/esco/occupation/00030d09-2b3a-4efd-87cc-c4ea39d27c34", "Technical Director (ESCO)"),
    ]

    report_md = f"""# Skill Graph Construction Sanity Check (Milestone M4)
*Generated by CareerSynk Impact — Stage A Graph Pipeline*

## 1. Graph Structural Summary
- **Merge Confidence Threshold**: {summary_metrics['merge_confidence_threshold']}
- **Total Nodes**: {summary_metrics['total_nodes']:,}
  - Occupations: {summary_metrics['nodes_by_type'].get('occupation', 0):,}
  - Skills: {summary_metrics['nodes_by_type'].get('skill', 0):,}
- **Total Edges**: {summary_metrics['total_edges']:,}
  - `requires` (Occupation -> Skill): {summary_metrics['edges_by_type'].get('requires', 0):,}
  - `possibly_related` (Skill <-> Skill): {summary_metrics['edges_by_type'].get('possibly_related', 0):,}

---

## 2. Sample Occupations: Top-10 Required Skills by Weight

"""
    for node_id, label in sample_queries:
        if not G.has_node(node_id):
            # Fallback to first available occupation of that source if exact ID differs
            prefix = node_id.split(":")[1]
            candidates = [n for n in G.nodes if n.startswith(f"occ:{prefix}")]
            if candidates:
                node_id = candidates[0]
                label = f"{G.nodes[node_id].get('label', node_id)} ({prefix.upper()})"
            else:
                continue

        occ_label = G.nodes[node_id].get("label", node_id)
        report_md += f"### {label}\n"
        report_md += f"- **Node ID**: `{node_id}`\n"
        report_md += "| Rank | Required Skill | Edge Weight | Edge Type | Skill Source |\n"
        report_md += "| :--- | :--- | :--- | :--- | :--- |\n"

        # Find all 'requires' neighbors
        neighbors = []
        for neighbor in G.neighbors(node_id):
            edge_data = G.get_edge_data(node_id, neighbor)
            if edge_data.get("edge_type") == "requires":
                skill_label = G.nodes[neighbor].get("label", neighbor)
                skill_source = G.nodes[neighbor].get("source", "unknown")
                weight = edge_data.get("weight", 0.0)
                neighbors.append((skill_label, weight, edge_data.get("edge_type"), skill_source))

        # Sort descending by weight
        neighbors.sort(key=lambda x: x[1], reverse=True)

        for rank, (s_label, s_weight, s_type, s_src) in enumerate(neighbors[:10], start=1):
            report_md += f"| {rank} | {s_label} | {s_weight:.4f} | `{s_type}` | `{s_src}` |\n"
        report_md += "\n"

    report_md += """---

## 3. Explicit Known Limitations & Prerequisite Gap for M5
> [!WARNING]
> **No Genuine Prerequisite Ordering Signal in Current Data**:
> - The current graph connects Occupations to Skills via observed requirements (`requires`), and Skills to Skills via semantic embedding similarity (`possibly_related`).
> - **The current raw datasets (O*NET 31.0 and ESCO v1.1) DO NOT contain directed prerequisite sequences** (e.g. "Skill A must be learned before Skill B").
> - **Recommendation for Milestone M5**: To model actionable counterfactual skill transitions, M5 must define a principled prerequisite proxy. Viable options include:
>   1. **Skill-to-Skill Co-occurrence Density**: Directional graph flow derived from junior vs. senior occupation requirement distributions.
>   2. **Taxonomic Depth / Complexity Scaling**: Lower-level foundational skills (e.g., O*NET Foundational Essential Skills or ESCO Level-1 clusters) preceding domain-specific technical specializations.
>   3. **Transversal-to-Specialized Hierarchy DAGs**: Edges directed from Transversal Competences to specialized task skills.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    occ_skills_file = project_root / "data" / "processed" / "occupation_skills.parquet"
    crosswalk_file = project_root / "data" / "processed" / "skill_crosswalk.parquet"
    out_graph = project_root / "data" / "processed" / "skill_graph.pkl"
    sanity_file = project_root / "data" / "processed" / "graph_sanity_check.md"

    print("=" * 60)
    print("Building Skill & Occupation Graph (M4)")
    print("=" * 60)
    G, summary = build_skill_graph(
        occupation_skills_path=occ_skills_file,
        crosswalk_path=crosswalk_file,
        merge_threshold=MERGE_CONFIDENCE_THRESHOLD,
        output_graph_path=out_graph,
        sanity_check_path=sanity_file,
    )
    print(f"Graph Construction Completed:")
    print(f"- Total Nodes: {summary['total_nodes']:,} (Occupations: {summary['nodes_by_type']['occupation']:,}, Skills: {summary['nodes_by_type']['skill']:,})")
    print(f"- Total Edges: {summary['total_edges']:,} (Requires: {summary['edges_by_type']['requires']:,}, Possibly Related: {summary['edges_by_type']['possibly_related']:,})")
    print(f"- Saved Graph to: {out_graph}")
    print(f"- Saved Sanity Check Report to: {sanity_file}")
