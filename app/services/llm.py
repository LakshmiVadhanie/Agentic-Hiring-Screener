from __future__ import annotations

import json
import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import get_settings

logger = logging.getLogger(__name__)

_llm_instance = None


def get_llm() -> ChatMistralAI:
    """Return a singleton Mistral LLM client."""
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()
        _llm_instance = ChatMistralAI(
            model="mistral-large-latest",
            api_key=settings.mistral_api_key,
            temperature=0.1,
            max_tokens=2048,
        )
    return _llm_instance


def evaluate_candidate(
    resume_text: str,
    job_title: str,
    job_description: str,
    required_skills: list[str],
    preferred_skills: list[str],
    min_experience_years: int,
) -> dict:
    """
    Use Mistral to evaluate a single candidate against a job description.
    Returns a structured assessment with scores and reasoning.
    """
    llm = get_llm()

    system_prompt = """You are an expert technical recruiter. Evaluate the candidate's resume
against the job requirements. Return ONLY valid JSON with these fields:
{
    "fit_score": <float 0-1>,
    "skills_matched": [<list of matched skills>],
    "skills_missing": [<list of missing required skills>],
    "experience_summary": "<1-2 sentence summary of relevant experience>",
    "recommendation": "<shortlist | review | reject with brief reason>"
}
Be precise and fair. Score based on evidence in the resume, not assumptions."""

    user_prompt = f"""## Job Details
- Title: {job_title}
- Description: {job_description}
- Required Skills: {', '.join(required_skills)}
- Preferred Skills: {', '.join(preferred_skills)}
- Minimum Experience: {min_experience_years} years

## Candidate Resume
{resume_text[:4000]}

Evaluate this candidate. Return JSON only."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = response.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]

        return json.loads(content)

    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response, using fallback scores")
        return {
            "fit_score": 0.0,
            "skills_matched": [],
            "skills_missing": required_skills,
            "experience_summary": "Could not parse LLM evaluation",
            "recommendation": "review",
        }
    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return {
            "fit_score": 0.0,
            "skills_matched": [],
            "skills_missing": list(required_skills),
            "experience_summary": f"LLM evaluation unavailable: {type(e).__name__}",
            "recommendation": "review",
        }