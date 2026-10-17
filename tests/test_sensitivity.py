"""Tests for sensitivity analysis module.

Verifies that:
- compute_jaccard_similarity computes bounded, accurate set overlap in [0.0, 1.0].
- run_sensitivity_analysis executes across all 3 alpha values [0.2, 0.5, 0.8].
- Stability metrics are generated and stay bounded.
- Uses a synthetic graph fixture to ensure fast execution.
"""

import pytest
import networkx as nx

from src.models.sensitivity_analysis import (
    compute_jaccard_similarity,
    run_sensitivity_analysis,
    ALPHA_VALUES,
)


def test_jaccard_similarity_bounds_and_values():
    """Verify Jaccard calculation on known sets."""
    set_a = {"python", "sql", "git"}
    set_b = {"python", "sql", "docker"}
    # intersection: 2 (python, sql), union: 4 (python, sql, git, docker) -> 0.5
    j = compute_jaccard_similarity(set_a, set_b)
    assert j == 0.5
    assert 0.0 <= j <= 1.0

    # Identical sets
    assert compute_jaccard_similarity(set_a, set_a) == 1.0

    # Disjoint sets
    assert compute_jaccard_similarity({"a"}, {"b"}) == 0.0

    # Empty sets
    assert compute_jaccard_similarity(set(), set()) == 1.0


@pytest.fixture
def synthetic_graph():
    """Build a lightweight synthetic graph fixture with 1 occupation and 3 skills."""
    g = nx.Graph()
    occ_id = "occ:synthetic:01"
    g.add_node(occ_id, node_type="occupation", label="Synthetic Dev")

    skills = ["Python", "Algorithms", "Databases"]
    for s in skills:
        s_id = f"skill:{s.lower()}"
        g.add_node(s_id, node_type="skill", label=s)
        g.add_edge(occ_id, s_id, edge_type="requires", weight=0.8)

    return g


def test_sensitivity_analysis_synthetic_run(synthetic_graph):
    """Verify run_sensitivity_analysis runs across all 3 alphas on synthetic fixture."""
    test_profiles = {
        "TestProfile": {
            "name": "Candidate A",
            "skills": ["Python"],
        }
    }
    test_occupations = [
        {"id": "occ:synthetic:01", "title": "Synthetic Dev"}
    ]

    results = run_sensitivity_analysis(
        profiles=test_profiles,
        target_occupations=test_occupations,
        model_artifact=None,
        graph=synthetic_graph,
        alphas=ALPHA_VALUES,
        top_k=2,
    )

    assert "occ:synthetic:01" in results
    prof_res = results["occ:synthetic:01"]["profiles"]["TestProfile"]
    assert 0.2 in prof_res["rankings"]
    assert 0.5 in prof_res["rankings"]
    assert 0.8 in prof_res["rankings"]

    # Missing skills should be Algorithms and Databases
    for alpha in ALPHA_VALUES:
        ranked = prof_res["rankings"][alpha]
        assert len(ranked) == 2
        assert "Python" not in ranked
        assert "Algorithms" in ranked
        assert "Databases" in ranked

    assert 0.0 <= prof_res["jaccard_02_08"] <= 1.0
    assert 0.0 <= prof_res["jaccard_05_08"] <= 1.0
    assert prof_res["jaccard_02_08"] == 1.0  # Under no model artifact, weights are equal
