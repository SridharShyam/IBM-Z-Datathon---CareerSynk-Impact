"""FastAPI REST Service for CareerSynk Impact (Milestone M7).

Exposes CPU-optimized, synchronous readiness scoring and counterfactual skill-gap
ranking designed for deployment on IBM LinuxONE (s390x) and standard Linux/amd64 containers.

Endpoints:
  - GET  /health: System health and artifact loading verification.
  - POST /api/v1/readiness: Evaluates candidate career readiness score [0.0, 1.0].
  - POST /api/v1/counterfactual-ranking: Simulates skill interventions and ranks gaps by delta.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import pickle
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from src.api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    HealthResponse,
    RankedSkillGap,
    ReadinessRequest,
    ReadinessResponse,
)
from src.models.counterfactual_ranking import rank_skill_gaps
from src.models.readiness_score import compute_readiness, MANDATORY_DISCLAIMER

# Global state loaded once on startup
APP_STATE: Dict[str, Optional[object]] = {
    "graph": None,
    "model_artifact": None,
}


def resolve_occupation_node(graph: nx.Graph, raw_id: str) -> str:
    """Resolve user-supplied occupation identifier to canonical graph node ID."""
    if graph.has_node(raw_id):
        return raw_id

    # Try standard prefix patterns
    for prefix in ["occ:onet:", "occ:esco:"]:
        candidate = f"{prefix}{raw_id}"
        if graph.has_node(candidate):
            return candidate

    # Search by suffix
    for node in graph.nodes:
        if node.endswith(f":{raw_id}"):
            return node

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=(
            f"Occupation ID '{raw_id}' was not found in the graph. "
            "Please provide a valid O*NET SOC Code (e.g. '15-1252.00') "
            "or full node identifier (e.g. 'occ:onet:15-1252.00')."
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager loading graph and model artifacts into memory once at startup."""
    project_root = Path(__file__).resolve().parent.parent.parent
    graph_path = project_root / "data" / "processed" / "skill_graph.pkl"
    model_path = project_root / "data" / "processed" / "baseline_relevance_model.pkl"

    print("=" * 60)
    print("CareerSynk Impact API: Initializing In-Memory Engine...")
    print("=" * 60)

    if graph_path.is_file():
        with open(graph_path, "rb") as f:
            APP_STATE["graph"] = pickle.load(f)
        print(f"-> Loaded skill graph: {APP_STATE['graph'].number_of_nodes():,} nodes, {APP_STATE['graph'].number_of_edges():,} edges.")
    else:
        print(f"-> WARNING: Graph artifact not found at {graph_path}")

    if model_path.is_file():
        with open(model_path, "rb") as f:
            APP_STATE["model_artifact"] = pickle.load(f)
        print("-> Loaded Stage A baseline relevance model artifact.")
    else:
        print(f"-> WARNING: Model artifact not found at {model_path}")

    print("Engine initialization complete. Ready for inference.\n")
    yield
    # Cleanup on shutdown
    APP_STATE["graph"] = None
    APP_STATE["model_artifact"] = None


app = FastAPI(
    title="CareerSynk Impact API",
    description=(
        "Skill-Gap Counterfactual Ranking and Career Readiness System. "
        "Built for IBM Z Datathon 2026 (Team ORDER 404, Tech for Good). "
        "Optimized for IBM LinuxONE CPU inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for demo UI / external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """System health check and artifact verification."""
    graph: Optional[nx.Graph] = APP_STATE.get("graph")
    model_artifact = APP_STATE.get("model_artifact")

    return HealthResponse(
        status="ok" if (graph is not None and model_artifact is not None) else "degraded",
        model_loaded=(model_artifact is not None),
        graph_loaded=(graph is not None),
        total_nodes=graph.number_of_nodes() if graph else 0,
        total_edges=graph.number_of_edges() if graph else 0,
    )


@app.post("/api/v1/readiness", response_model=ReadinessResponse, tags=["Scoring"])
def post_readiness(payload: ReadinessRequest) -> ReadinessResponse:
    """Calculate candidate readiness score for a target occupation."""
    graph: Optional[nx.Graph] = APP_STATE.get("graph")
    model_artifact = APP_STATE.get("model_artifact")

    if graph is None or model_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine artifacts are not loaded in memory.",
        )

    canonical_occ_node = resolve_occupation_node(graph, payload.occupation_id)
    occ_title = graph.nodes[canonical_occ_node].get("label", payload.occupation_id)

    baseline_score, breakdown = compute_readiness(
        candidate_skills=set(payload.candidate_skills),
        occupation_id=canonical_occ_node,
        model_artifact=model_artifact,
        graph=graph,
    )

    return ReadinessResponse(
        occupation_id=canonical_occ_node,
        occupation_title=occ_title,
        baseline_score=baseline_score,
        skills_evaluated_count=len(breakdown),
        disclaimer=MANDATORY_DISCLAIMER,
    )


@app.post("/api/v1/counterfactual-ranking", response_model=CounterfactualResponse, tags=["Ranking"])
def post_counterfactual_ranking(payload: CounterfactualRequest) -> CounterfactualResponse:
    """Simulate single-skill interventions and rank gaps by predicted readiness delta."""
    graph: Optional[nx.Graph] = APP_STATE.get("graph")
    model_artifact = APP_STATE.get("model_artifact")

    if graph is None or model_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine artifacts are not loaded in memory.",
        )

    canonical_occ_node = resolve_occupation_node(graph, payload.occupation_id)
    occ_title = graph.nodes[canonical_occ_node].get("label", payload.occupation_id)

    baseline_score, _ = compute_readiness(
        candidate_skills=set(payload.candidate_skills),
        occupation_id=canonical_occ_node,
        model_artifact=model_artifact,
        graph=graph,
    )

    ranked_df = rank_skill_gaps(
        candidate_skills=set(payload.candidate_skills),
        occupation_id=canonical_occ_node,
        model_artifact=model_artifact,
        graph=graph,
        top_n=payload.top_n,
    )

    ranked_gaps = [
        RankedSkillGap(
            skill_name=row["skill_name"],
            hypothetical_score=row["hypothetical_score"],
            predicted_delta=row["predicted_delta"],
            requirement_weight=row["requirement_weight"],
            explanation=row["explanation_text"],
        )
        for _, row in ranked_df.iterrows()
    ]

    return CounterfactualResponse(
        occupation_id=canonical_occ_node,
        occupation_title=occ_title,
        baseline_score=baseline_score,
        ranked_gaps=ranked_gaps,
        disclaimer=MANDATORY_DISCLAIMER,
    )
