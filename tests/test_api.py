"""Integration tests for FastAPI application endpoints.

Tests all 3 endpoints:
- GET /health
- POST /api/v1/readiness
- POST /api/v1/counterfactual-ranking
Validates happy paths, 404 for unknown occupation, and 422 for malformed requests.
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.readiness_score import MANDATORY_DISCLAIMER

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRAPH_PATH = os.path.join(BASE_DIR, "data", "processed", "skill_graph.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "data", "processed", "baseline_relevance_model.pkl")

artifacts_present = os.path.exists(GRAPH_PATH) and os.path.exists(MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    """Create TestClient with lifespan context entered."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.skipif(not artifacts_present, reason="M4/M5 artifacts missing")
def test_health_endpoint(client):
    """Test GET /health returns 200 with model and graph loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["graph_loaded"] is True


@pytest.mark.skipif(not artifacts_present, reason="M4/M5 artifacts missing")
def test_readiness_endpoint_happy_path(client):
    """Test POST /api/v1/readiness returns valid score and non-causal disclaimer."""
    payload = {
        "candidate_skills": ["python", "software engineering", "git", "sql"],
        "occupation_id": "occ:onet:15-1252.00",
    }
    response = client.post("/api/v1/readiness", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["occupation_id"] == "occ:onet:15-1252.00"
    assert 0.0 <= data["baseline_score"] <= 1.0
    assert data["disclaimer"] == MANDATORY_DISCLAIMER


@pytest.mark.skipif(not artifacts_present, reason="M4/M5 artifacts missing")
def test_readiness_unknown_occupation_404(client):
    """Test POST /api/v1/readiness returns 404 for unknown occupation_id."""
    payload = {
        "candidate_skills": ["python"],
        "occupation_id": "occ:unknown:99-9999.99",
    }
    response = client.post("/api/v1/readiness", json=payload)
    assert response.status_code == 404
    detail = response.json().get("detail", "")
    assert "was not found in the graph" in detail.lower()


def test_readiness_validation_error_422(client):
    """Test POST /api/v1/readiness returns 422 on invalid schema."""
    payload = {
        "candidate_skills": "not-a-list",
        "occupation_id": 12345,
    }
    response = client.post("/api/v1/readiness", json=payload)
    assert response.status_code == 422


@pytest.mark.skipif(not artifacts_present, reason="M4/M5 artifacts missing")
def test_counterfactual_ranking_happy_path(client):
    """Test POST /api/v1/counterfactual-ranking returns ranked skill gaps."""
    payload = {
        "candidate_skills": ["python", "git"],
        "occupation_id": "occ:onet:15-1252.00",
        "top_n": 5,
    }
    response = client.post("/api/v1/counterfactual-ranking", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["occupation_id"] == "occ:onet:15-1252.00"
    assert 0.0 <= data["baseline_score"] <= 1.0
    assert data["disclaimer"] == MANDATORY_DISCLAIMER
    assert isinstance(data["ranked_gaps"], list)
    assert len(data["ranked_gaps"]) <= 5
    if data["ranked_gaps"]:
        gap = data["ranked_gaps"][0]
        assert "skill_name" in gap
        assert "predicted_delta" in gap
        assert "requirement_weight" in gap
        assert "explanation" in gap
        assert gap["predicted_delta"] >= 0.0


@pytest.mark.skipif(not artifacts_present, reason="M4/M5 artifacts missing")
def test_counterfactual_unknown_occupation_404(client):
    """Test POST /api/v1/counterfactual-ranking returns 404 for unknown occupation."""
    payload = {
        "candidate_skills": ["python"],
        "occupation_id": "nonexistent_occ_id",
        "top_n": 5,
    }
    response = client.post("/api/v1/counterfactual-ranking", json=payload)
    assert response.status_code == 404
    assert "not found in the graph" in response.json().get("detail", "").lower()
