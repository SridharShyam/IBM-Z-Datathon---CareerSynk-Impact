"""Unit and integration tests for counterfactual skill-gap ranking (Milestone M5)."""

from pathlib import Path
import networkx as nx
import pandas as pd
import pytest

from src.models.counterfactual_ranking import rank_skill_gaps


class TestCounterfactualRankingSynthetic:
    """Synthetic unit tests for counterfactual simulation, delta ordering, and exclusion of owned skills."""

    def test_ranking_excludes_already_owned_skills_and_sorts_by_delta(self):
        G = nx.Graph()
        occ_id = "occ:synthetic:dev_role"
        G.add_node(occ_id, node_type="occupation", label="Developer")

        # Three skills:
        # Skill 1: Owned by candidate
        # Skill 2: Missing, high weight (0.90) -> should yield highest delta
        # Skill 3: Missing, low weight (0.20) -> should yield lower delta
        G.add_node("skill:s1", node_type="skill", label="Python")
        G.add_node("skill:s2", node_type="skill", label="Databases")
        G.add_node("skill:s3", node_type="skill", label="Typescript")

        G.add_edge(occ_id, "skill:s1", edge_type="requires", weight=0.8)
        G.add_edge(occ_id, "skill:s2", edge_type="requires", weight=0.9)
        G.add_edge(occ_id, "skill:s3", edge_type="requires", weight=0.2)

        candidate_skills = {"Python"}

        ranked_df = rank_skill_gaps(
            candidate_skills=candidate_skills,
            occupation_id=occ_id,
            graph=G,
            top_n=5,
        )

        assert isinstance(ranked_df, pd.DataFrame)
        # Expected schema
        expected_cols = [
            "skill_name", "baseline_score", "hypothetical_score",
            "predicted_delta", "requirement_weight", "explanation_text"
        ]
        assert list(ranked_df.columns) == expected_cols

        # 1. Python must be excluded because candidate already holds it
        assert "Python" not in ranked_df["skill_name"].values
        assert len(ranked_df) == 2

        # 2. Databases (weight 0.9) must rank above Typescript (weight 0.2)
        assert ranked_df.iloc[0]["skill_name"] == "Databases"
        assert ranked_df.iloc[1]["skill_name"] == "Typescript"

        # 3. Predicted deltas must be strictly positive
        assert ranked_df.iloc[0]["predicted_delta"] > ranked_df.iloc[1]["predicted_delta"]
        assert ranked_df.iloc[0]["predicted_delta"] > 0.0

        # 4. Explanations must be non-empty and descriptive
        assert len(ranked_df.iloc[0]["explanation_text"]) > 20

    def test_full_candidate_returns_empty_gap_ranking(self):
        G = nx.Graph()
        occ_id = "occ:synthetic:role_done"
        G.add_node(occ_id, node_type="occupation", label="Role Done")
        G.add_node("skill:s1", node_type="skill", label="Skill One")
        G.add_edge(occ_id, "skill:s1", edge_type="requires", weight=0.5)

        # Candidate possesses all required skills
        ranked_df = rank_skill_gaps(
            candidate_skills={"Skill One"},
            occupation_id=occ_id,
            graph=G,
        )

        assert len(ranked_df) == 0


@pytest.mark.integration
def test_real_counterfactual_ranking_integration():
    """Integration test checking that counterfactual ranking executes against real artifacts."""
    graph_path = Path("data/processed/skill_graph.pkl")
    model_path = Path("data/processed/baseline_relevance_model.pkl")

    if not (graph_path.is_file() and model_path.is_file()):
        pytest.skip("Real graph or model artifact not present in data/processed/")

    import pickle
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    ranked_df = rank_skill_gaps(
        candidate_skills={"Programming", "Mathematics"},
        occupation_id="occ:onet:15-1252.00",
        model_artifact=model_artifact,
        graph=graph,
        top_n=5,
    )

    assert len(ranked_df) == 5
    assert "Programming" not in ranked_df["skill_name"].values
    assert (ranked_df["predicted_delta"] > 0.0).all()
