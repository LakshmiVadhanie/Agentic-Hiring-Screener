import logging
from app.agents.state import ScreeningState
from app.services.embedding import search_similar_candidates
from app.services.llm import evaluate_candidate
from app.models.schemas import CandidateScore, ScreeningStatus

logger = logging.getLogger(__name__)

# ── Weights for final score ──
SIMILARITY_WEIGHT = 0.2
LLM_WEIGHT = 0.8

def retriever_node(state: ScreeningState) -> dict:
    """
    Node 1: Retrieve top-K semantically similar resumes from ChromaDB.
    Uses MPNet embeddings to match job description against indexed resumes.
    """
    job = state["job"]
    query = f"{job.title} {job.description} {' '.join(job.required_skills)}"

    logger.info(f"Retrieving top {state['top_k']} candidates for: {job.title}")

    candidates = search_similar_candidates(
        job_description=query,
        top_k=100,  # Over-fetch significantly so the LLM evaluator sees a broad pool
        threshold=state["threshold"] * 0.5,  # Lower threshold for retrieval stage
    )

    logger.info(f"Retrieved {len(candidates)} candidates from vector store")
    return {"retrieved_candidates": candidates, "total_screened": len(candidates)}


def evaluator_node(state: ScreeningState) -> dict:
    """
    Node 2: Use Mistral LLM to deeply evaluate each retrieved candidate.
    Produces structured scores per candidate.
    """
    job = state["job"]
    retrieved = state["retrieved_candidates"]

    if not retrieved:
        logger.warning("No candidates to evaluate")
        return {"evaluated_candidates": []}

    evaluated = []
    for candidate in retrieved:
        logger.info(f"Evaluating candidate: {candidate['id']}")

        assessment = evaluate_candidate(
            resume_text=candidate["document"],
            job_title=job.title,
            job_description=job.description,
            required_skills=job.required_skills,
            preferred_skills=job.preferred_skills,
            min_experience_years=job.min_experience_years,
        )

        similarity = candidate["similarity"]
        llm_score = assessment.get("fit_score", 0.0)
        combined = (SIMILARITY_WEIGHT * similarity) + (LLM_WEIGHT * llm_score)

        recommendation = assessment.get("recommendation", "review")
        if "shortlist" in recommendation.lower():
            status = ScreeningStatus.SHORTLISTED
        elif "reject" in recommendation.lower():
            status = ScreeningStatus.REJECTED
        else:
            status = ScreeningStatus.REVIEW

        scored = CandidateScore(
            candidate_id=candidate["id"],
            candidate_name=candidate["metadata"].get("candidate_name", "Unknown"),
            similarity_score=similarity,
            llm_score=llm_score,
            combined_score=round(combined, 4),
            skills_matched=assessment.get("skills_matched", []),
            skills_missing=assessment.get("skills_missing", []),
            experience_summary=assessment.get("experience_summary", ""),
            recommendation=assessment.get("recommendation", ""),
            status=status,
        )
        evaluated.append(scored)

    logger.info(f"Evaluated {len(evaluated)} candidates")
    return {"evaluated_candidates": evaluated}


def ranker_node(state: ScreeningState) -> dict:
    """
    Node 3: Rank evaluated candidates by combined score and apply threshold.
    Produces the final shortlist.
    """
    evaluated = state["evaluated_candidates"]
    threshold = state["threshold"]
    top_k = state["top_k"]

    # Sort by combined score descending
    ranked = sorted(evaluated, key=lambda c: c.combined_score, reverse=True)

    # Apply threshold and limit
    shortlisted = [
        c for c in ranked
        if c.combined_score >= threshold
    ][:top_k]

    logger.info(
        f"Ranked {len(evaluated)} candidates, "
        f"shortlisted {len(shortlisted)} (threshold={threshold})"
    )
    return {"shortlisted": shortlisted}


def should_continue(state: ScreeningState) -> str:
    """Conditional edge: skip evaluation if no candidates retrieved."""
    if not state.get("retrieved_candidates"):
        return "no_candidates"
    return "evaluate"