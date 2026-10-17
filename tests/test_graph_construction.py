"""Unit and integration tests for heterogeneous skill graph construction (Milestone M4)."""

from pathlib import Path
import networkx as nx
import pandas as pd
import pytest

from src.graph.build_skill_graph import (
    build_skill_graph,
    MERGE_CONFIDENCE_THRESHOLD,
    ESCO_RELATION_WEIGHTS,
)


class TestGraphConstructionFixtures:
    """Synthetic unit tests verifying node/edge typing, weight mapping, and unification rules."""

    def test_synthetic_graph_building_and_typing(self, tmp_path: Path):
        # 1. Synthetic occupation_skills dataframe
        occ_skills_df = pd.DataFrame([
            {
                "occupation_id": "11-1011.00",
                "occupation_title": "Chief Executives",
                "skill_name": "Active Listening",
                "relation_or_weight": "4.5",  # O*NET weight
                "source": "onet",
            },
            {
                "occupation_id": "http://data.europa.eu/esco/occupation/sample-1",
                "occupation_title": "Software Developer",
                "skill_name": "listen actively",
                "relation_or_weight": "essential",  # ESCO weight
                "source": "esco",
            },
            {
                "occupation_id": "http://data.europa.eu/esco/occupation/sample-1",
                "occupation_title": "Software Developer",
                "skill_name": "underwater basketweaving",
                "relation_or_weight": "optional",  # ESCO weight
                "source": "esco",
            },
        ])

        # 2. Synthetic crosswalk dataframe
        crosswalk_df = pd.DataFrame([
            {
                "onet_skill_name": "Active Listening",
                "esco_skill_name": "listen actively",
                "embedding_score": 0.85,  # >= 0.80 -> unified
                "hierarchy_supported": True,
                "included_in_topk": True,
            },
            {
                "onet_skill_name": "Active Listening",
                "esco_skill_name": "underwater basketweaving",
                "embedding_score": 0.40,  # < 0.80 -> separate linked
                "hierarchy_supported": False,
                "included_in_topk": False,
            },
        ])

        occ_skills_file = tmp_path / "occ_skills.parquet"
        crosswalk_file = tmp_path / "crosswalk.parquet"
        out_graph_file = tmp_path / "graph.pkl"
        sanity_check_file = tmp_path / "sanity.md"

        occ_skills_df.to_parquet(occ_skills_file)
        crosswalk_df.to_parquet(crosswalk_file)

        G, summary = build_skill_graph(
            occupation_skills_path=occ_skills_file,
            crosswalk_path=crosswalk_file,
            merge_threshold=0.80,
            output_graph_path=out_graph_file,
            sanity_check_path=sanity_check_file,
        )

        assert isinstance(G, nx.Graph)
        assert summary["total_nodes"] > 0
        assert summary["total_edges"] > 0
        assert out_graph_file.is_file()
        assert sanity_check_file.is_file()

        # Check unification
        # 'Active Listening' and 'listen actively' should merge into 'skill:unified:Active Listening'
        unified_id = "skill:unified:Active Listening"
        assert G.has_node(unified_id)
        assert G.nodes[unified_id]["source"] == "unified"

        # Check separate nodes linked for low score
        basket_id = "skill:esco:underwater basketweaving"
        assert G.has_node(basket_id)
        assert G.has_edge(unified_id, basket_id)
        assert G.get_edge_data(unified_id, basket_id)["edge_type"] == "possibly_related"
        assert G.get_edge_data(unified_id, basket_id)["weight"] == 0.40

        # Check weight normalization
        # O*NET 4.5 / 5.0 = 0.90
        onet_occ_id = "occ:onet:11-1011.00"
        assert G.has_edge(onet_occ_id, unified_id)
        assert G.get_edge_data(onet_occ_id, unified_id)["weight"] == 0.90

        # ESCO essential = 1.0, optional = 0.5
        esco_occ_id = "occ:esco:http://data.europa.eu/esco/occupation/sample-1"
        assert G.has_edge(esco_occ_id, unified_id)
        assert G.get_edge_data(esco_occ_id, unified_id)["weight"] == 1.0
        assert G.has_edge(esco_occ_id, basket_id)
        assert G.get_edge_data(esco_occ_id, basket_id)["weight"] == 0.5


@pytest.mark.integration
def test_real_data_skill_graph_integration():
    """Integration test checking that the generated real graph exists and has expected topology."""
    graph_path = Path("data/processed/skill_graph.pkl")
    if not graph_path.is_file():
        pytest.skip("Processed skill_graph.pkl not present.")

    import pickle
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 10000
    assert G.number_of_edges() > 100000

    # Ensure Chief Executives exists and has requires edges
    assert G.has_node("occ:onet:11-1011.00")
    neighbors = list(G.neighbors("occ:onet:11-1011.00"))
    assert len(neighbors) == 35  # All 35 O*NET skills required with positive weights
