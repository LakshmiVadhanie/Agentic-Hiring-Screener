import logging
import time
from langgraph.graph import StateGraph, END
from app.agents.state import ScreeningState
from app.agents.nodes import (
    retriever_node,
    evaluator_node,
    ranker_node,
    should_continue,
)
from app.models.schemas import JobDescription, ScreeningResult

logger = logging.getLogger(__name__)


def build_screening_graph() -> StateGraph:
    """
    Construct the LangGraph multi-agent screening pipeline.

    Flow:
        retrieve -> (check candidates) -> evaluate -> rank -> END
                                      |
                                      +-> END (no candidates)
    """
    graph = StateGraph(ScreeningState)

    # Add nodes
    graph.add_node("retrieve", retriever_node)
    graph.add_node("evaluate", evaluator_node)
    graph.add_node("rank", ranker_node)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Conditional routing after retrieval
    graph.add_conditional_edges(
        "retrieve",
        should_continue,
        {
            "evaluate": "evaluate",
            "no_candidates": END,
        },
    )

    # Linear flow: evaluate -> rank -> end
    graph.add_edge("evaluate", "rank")
    graph.add_edge("rank", END)

    return graph.compile()


# Compile once at module level
screening_pipeline = build_screening_graph()


def run_screening(
    job: JobDescription,
    top_k: int = 10,
    threshold: float = 0.5,
) -> ScreeningResult:
    """
    Execute the full screening pipeline for a job description.
    Returns a structured ScreeningResult.
    """
    start = time.time()

    initial_state: ScreeningState = {
        "job": job,
        "top_k": top_k,
        "threshold": threshold,
        "retrieved_candidates": [],
        "evaluated_candidates": [],
        "shortlisted": [],
        "total_screened": 0,
        "error": "",
    }

    logger.info(f"Starting screening pipeline for: {job.title}")
    final_state = screening_pipeline.invoke(initial_state)
    elapsed = round(time.time() - start, 2)

    result = ScreeningResult(
        job_title=job.title,
        total_candidates_screened=final_state.get("total_screened", 0),
        shortlisted=final_state.get("shortlisted", []),
        screening_time_seconds=elapsed,
    )

    logger.info(
        f"Screening complete: {result.total_candidates_screened} screened, "
        f"{len(result.shortlisted)} shortlisted in {elapsed}s"
    )
    return result