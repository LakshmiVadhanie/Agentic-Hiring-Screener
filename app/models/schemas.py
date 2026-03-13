"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Job Description ──────────────────────────────────────────────────────────


class JobDescription(BaseModel):
    """Schema for a job description used in screening."""

    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Full job description text")
    required_skills: list[str] = Field(
        ..., description="List of required skills"
    )
    preferred_skills: list[str] = Field(
        default_factory=list, description="List of preferred/nice-to-have skills"
    )
    min_experience_years: int = Field(
        default=0, description="Minimum years of experience required"
    )


# ── Screening Status ─────────────────────────────────────────────────────────


class ScreeningStatus(str, Enum):
    """Status assigned to a candidate after evaluation."""

    SHORTLISTED = "shortlisted"
    REVIEW = "review"
    REJECTED = "rejected"


# ── Candidate Score ──────────────────────────────────────────────────────────


class CandidateScore(BaseModel):
    """A single candidate's screening result with scores."""

    candidate_id: str = ""
    candidate_name: str = ""
    similarity_score: float = 0.0
    llm_score: float = 0.0
    combined_score: float = 0.0
    skills_matched: list[str] = Field(default_factory=list)
    skills_missing: list[str] = Field(default_factory=list)
    experience_summary: str = ""
    recommendation: str = ""
    status: ScreeningStatus = ScreeningStatus.REVIEW


# ── Screening Request / Response ─────────────────────────────────────────────


class ScreeningRequest(BaseModel):
    """Request body for the screening endpoint."""

    job: JobDescription
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top candidates to return")
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum combined score threshold"
    )


class ScreeningResult(BaseModel):
    """Response from the screening pipeline."""

    job_title: str
    total_candidates_screened: int
    shortlisted: list[CandidateScore] = Field(default_factory=list)
    screening_time_seconds: float = 0.0


# ── Resume Indexing ──────────────────────────────────────────────────────────


class ResumeUpload(BaseModel):
    """Request body for indexing a resume."""

    candidate_name: str = Field(..., description="Candidate's full name")
    email: str = Field(default="", description="Candidate's email")
    resume_text: str = Field(..., description="Full resume text")
    source_filename: str = Field(default="Manual Upload", description="Source of the resume")


class ResumeIndexResponse(BaseModel):
    """Response after indexing a resume."""

    candidate_id: str
    candidate_name: str
    indexed: bool = True
    message: str
