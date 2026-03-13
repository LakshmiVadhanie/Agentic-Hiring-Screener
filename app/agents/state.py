"""Shared pipeline state for the LangGraph screening graph."""

from typing import TypedDict
from app.models.schemas import JobDescription, CandidateScore


class ScreeningState(TypedDict, total=False):
    """State passed through the LangGraph pipeline nodes.

    Attributes:
        job: The job description to screen candidates against.
        top_k: Number of top candidates to return.
        threshold: Minimum combined score to include a candidate.
        retrieved_candidates: Raw candidates from ChromaDB (list of dicts).
        evaluated_candidates: Candidates scored by the LLM.
        shortlisted: Final ranked and filtered candidate list.
        total_screened: Total candidates processed.
        error: Error message if any node fails.
    """

    job: JobDescription
    top_k: int
    threshold: float
    retrieved_candidates: list[dict]
    evaluated_candidates: list[CandidateScore]
    shortlisted: list[CandidateScore]
    total_screened: int
    error: str
