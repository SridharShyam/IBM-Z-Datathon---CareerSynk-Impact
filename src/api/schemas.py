"""Pydantic request and response schemas for CareerSynk Impact API (Milestone M7)."""

from typing import List, Optional
from pydantic import BaseModel, Field

from src.models.readiness_score import MANDATORY_DISCLAIMER


class ReadinessRequest(BaseModel):
    """Request payload for candidate career readiness scoring."""
    candidate_skills: List[str] = Field(
        ...,
        description="List of verified skills currently possessed by the candidate.",
        json_schema_extra={"example": ["Programming", "Mathematics", "Critical Thinking"]},
    )
    occupation_id: str = Field(
        ...,
        description="Target occupation identifier (e.g. '15-1252.00' or 'occ:onet:15-1252.00').",
        json_schema_extra={"example": "occ:onet:15-1252.00"},
    )


class ReadinessResponse(BaseModel):
    """Response payload containing computed career readiness score and disclaimer."""
    occupation_id: str = Field(..., description="Target occupation identifier.")
    occupation_title: str = Field(..., description="Human-readable title of the target occupation.")
    baseline_score: float = Field(..., description="Overall readiness score normalized between 0.0 and 1.0.")
    skills_evaluated_count: int = Field(..., description="Number of required skills evaluated for this occupation.")
    disclaimer: str = Field(
        default=MANDATORY_DISCLAIMER,
        description="Mandatory non-causal statistical modeling disclaimer.",
    )


class RankedSkillGap(BaseModel):
    """Individual skill gap recommendation ranked by predicted readiness delta."""
    skill_name: str = Field(..., description="Name of the missing skill recommended for acquisition.")
    hypothetical_score: float = Field(..., description="Projected readiness score upon acquiring this skill.")
    predicted_delta: float = Field(..., description="Marginal readiness score increase (+Delta).")
    requirement_weight: float = Field(..., description="Composite requirement weight of the skill in this occupation.")
    explanation: str = Field(..., description="Deterministic, data-traceable explanation for the recommendation.")


class CounterfactualRequest(BaseModel):
    """Request payload for counterfactual skill-gap ranking."""
    candidate_skills: List[str] = Field(
        ...,
        description="List of verified skills currently possessed by the candidate.",
        json_schema_extra={"example": ["Active Listening", "Speaking", "Coordination"]},
    )
    occupation_id: str = Field(
        ...,
        description="Target occupation identifier (e.g. '15-1252.00' or 'occ:onet:15-1252.00').",
        json_schema_extra={"example": "occ:onet:15-1252.00"},
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of top skill interventions to return.",
        json_schema_extra={"example": 10},
    )


class CounterfactualResponse(BaseModel):
    """Response payload containing ranked skill gaps with predicted deltas and disclaimer."""
    occupation_id: str = Field(..., description="Target occupation identifier.")
    occupation_title: str = Field(..., description="Human-readable title of the target occupation.")
    baseline_score: float = Field(..., description="Current baseline readiness score.")
    ranked_gaps: List[RankedSkillGap] = Field(
        default_factory=list,
        description="Ordered list of recommended skill gaps by predicted score delta.",
    )
    disclaimer: str = Field(
        default=MANDATORY_DISCLAIMER,
        description="Mandatory non-causal statistical modeling disclaimer.",
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="ok", description="Service health status.")
    model_loaded: bool = Field(..., description="Whether Stage A baseline ML model is loaded in memory.")
    graph_loaded: bool = Field(..., description="Whether heterogeneous skill graph is loaded in memory.")
