"""Unit and integration tests for Career Readiness Score formula (Milestone M5)."""

from pathlib import Path
import networkx as nx
import pytest

from src.models.readiness_score import (
    compute_skill_credit,
    compute_readiness,
    PARTIAL_CREDIT_DISCOUNT,
)


class TestReadinessScoreFormulaSynthetic:
    """Deterministic synthetic test cases validating formula behavior, bounds, and directional monotonicity."""

    def test_readiness_bounds_and_empty_candidate(self):
        # Construct synthetic graph
        G = nx.Graph()
        occ_id = "occ:synthetic:test_role"
        G.add_node(occ_id, node_type="occupation", label="Test Role")

        # Two skills: skill_high (weight 0.9), skill_low (weight 0.3)
        G.add_node("skill:s1", node_type="skill", label="High Priority Skill")
        G.add_node("skill:s2", node_type="skill", label="Low Priority Skill")

        G.add_edge(occ_id, "skill:s1", edge_type="requires", weight=0.9)
        G.add_edge(occ_id, "skill:s2", edge_type="requires", weight=0.3)

        # Case 1: Empty candidate skill profile -> readiness = 0.0
        score_empty, breakdown = compute_readiness(
            candidate_skills=set(),
            occupation_id=occ_id,
            graph=G,
        )
        assert score_empty == 0.0
        assert len(breakdown) == 2
        assert breakdown["High Priority Skill"]["credit"] == 0.0

        # Case 2: Full candidate skill profile -> readiness = 1.0
        score_full, _ = compute_readiness(
            candidate_skills={"High Priority Skill", "Low Priority Skill"},
            occupation_id=occ_id,
            graph=G,
        )
        assert score_full == 1.0

    def test_directional_sensitivity_high_vs_low_priority_skill(self):
        # Missing a high-weight skill must score LOWER than missing a low-weight skill
        G = nx.Graph()
        occ_id = "occ:synthetic:role_b"
        G.add_node(occ_id, node_type="occupation", label="Role B")

        G.add_node("skill:high", node_type="skill", label="Architecture")
        G.add_node("skill:low", node_type="skill", label="Documentation")

        G.add_edge(occ_id, "skill:high", edge_type="requires", weight=0.9)
        G.add_edge(occ_id, "skill:low", edge_type="requires", weight=0.1)

        # Candidate A has only high-priority skill
        score_cand_a, _ = compute_readiness(
            candidate_skills={"Architecture"},
            occupation_id=occ_id,
            graph=G,
        )

        # Candidate B has only low-priority skill
        score_cand_b, _ = compute_readiness(
            candidate_skills={"Documentation"},
            occupation_id=occ_id,
            graph=G,
        )

        assert score_cand_a > score_cand_b
        assert score_cand_a == pytest.approx(0.9 / (0.9 + 0.1), abs=1e-3)
        assert score_cand_b == pytest.approx(0.1 / (0.9 + 0.1), abs=1e-3)

    def test_partial_credit_via_graph_relation(self):
        # Candidate has an adjacent skill connected via 'related_to'
        G = nx.Graph()
        occ_id = "occ:synthetic:role_c"
        G.add_node(occ_id, node_type="occupation", label="Role C")

        target_skill = "skill:target"
        adjacent_skill = "skill:adjacent"
        G.add_node(target_skill, node_type="skill", label="Python")
        G.add_node(adjacent_skill, node_type="skill", label="R Programming")

        G.add_edge(occ_id, target_skill, edge_type="requires", weight=0.8)
        # Connect adjacent skill with 0.80 similarity
        G.add_edge(target_skill, adjacent_skill, edge_type="related_to", weight=0.80)

        # Candidate has R Programming (adjacent), not Python directly
        score_partial, breakdown = compute_readiness(
            candidate_skills={"R Programming"},
            occupation_id=occ_id,
            graph=G,
        )

        # Credit should be 0.80 * PARTIAL_CREDIT_DISCOUNT (0.50) = 0.40
        expected_credit = 0.80 * PARTIAL_CREDIT_DISCOUNT
        assert breakdown["Python"]["credit"] == pytest.approx(expected_credit, abs=1e-3)
        assert score_partial == pytest.approx(expected_credit, abs=1e-3)


@pytest.mark.integration
def test_real_readiness_integration():
    """Integration test verifying compute_readiness against real graph and model."""
    graph_path = Path("data/processed/skill_graph.pkl")
    model_path = Path("data/processed/baseline_relevance_model.pkl")

    if not (graph_path.is_file() and model_path.is_file()):
        pytest.skip("Graph or model artifact not present in data/processed/")

    import pickle
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)

    # Developer profile
    score, breakdown = compute_readiness(
        candidate_skills={"Programming", "Critical Thinking"},
        occupation_id="occ:onet:15-1252.00",
        model_artifact=model_artifact,
        graph=graph,
    )

    assert 0.0 < score < 1.0
    assert len(breakdown) == 35  # All 35 O*NET skills evaluated
    assert breakdown["Programming"]["credit"] == 1.0
