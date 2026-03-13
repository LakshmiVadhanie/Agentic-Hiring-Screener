"""API routes for the Agentic Hiring Screener."""

from __future__ import annotations

import logging
import re
from fastapi import APIRouter, HTTPException, UploadFile, File
from pypdf import PdfReader
import io

from app.models.schemas import (
    ScreeningRequest,
    ScreeningResult,
    ResumeUpload,
    ResumeIndexResponse,
)
from app.agents.graph import run_screening
from app.services.embedding import (
    index_resume,
    delete_from_index,
    get_collection_stats,
)
from app.services.database import (
    insert_candidate,
    get_candidate,
    find_candidate_by_email,
    delete_candidate,
    list_candidates,
    count_candidates,
    save_screening,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Screening ──────────────────────────────────────────────────────────────


@router.post("/screen", response_model=ScreeningResult)
async def screen_candidates(request: ScreeningRequest):
    """Run the full multi-agent screening pipeline."""
    try:
        result = run_screening(
            job=request.job,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        await save_screening(result.model_dump())
        return result
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        raise HTTPException(status_code=500, detail=f"Screening pipeline error: {e}")


# ── Resume: Parse Preview (no indexing) ────────────────────────────────────


@router.post("/resumes/parse-pdf")
async def parse_pdf_preview(file: UploadFile = File(...)):
    """
    Parse a PDF resume and return the extracted text, auto-detected name,
    and email WITHOUT saving or indexing. This is the preview step.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        candidate_name = _extract_name_from_resume(text)
        email = _extract_email_from_resume(text)

        # Check for duplicates
        existing = await find_candidate_by_email(email) if email else None

        return {
            "parsed_text": text,
            "candidate_name": candidate_name,
            "email": email,
            "pages": len(reader.pages),
            "char_count": len(text),
            "filename": file.filename,
            "duplicate": existing is not None,
            "existing_candidate": existing,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Resume: Index (after preview confirmation) ────────────────────────────


@router.post("/resumes/index", response_model=ResumeIndexResponse)
async def index_resume_text(upload: ResumeUpload):
    """Index a resume (plain text) into ChromaDB and save to MongoDB."""
    try:
        candidate_id = await insert_candidate({
            "candidate_name": upload.candidate_name,
            "email": upload.email,
            "resume_text": upload.resume_text,
            "source_filename": upload.source_filename,
        })
        index_resume(
            candidate_id=candidate_id,
            resume_text=upload.resume_text,
            metadata={
                "candidate_name": upload.candidate_name,
                "email": upload.email,
            },
        )
        return ResumeIndexResponse(
            candidate_id=candidate_id,
            candidate_name=upload.candidate_name,
            indexed=True,
            message="Resume indexed successfully",
        )
    except Exception as e:
        logger.error(f"Resume indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resumes/upload-pdf", response_model=ResumeIndexResponse)
async def upload_resume_pdf(file: UploadFile = File(...)):
    """Upload a PDF, auto-parse, and index in one step."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        candidate_name = _extract_name_from_resume(text)
        email = _extract_email_from_resume(text)

        candidate_id = await insert_candidate({
            "candidate_name": candidate_name,
            "email": email,
            "resume_text": text,
            "source_filename": file.filename,
        })
        index_resume(
            candidate_id=candidate_id,
            resume_text=text,
            metadata={"candidate_name": candidate_name, "email": email},
        )
        return ResumeIndexResponse(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            indexed=True,
            message=f"PDF parsed ({len(reader.pages)} pages, {len(text)} chars) — auto-detected: {candidate_name}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────


import spacy

# Load spaCy model globally (lazy loaded)
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not load spacy model: {e}")
            return None
    return _nlp

def _extract_name_from_resume(text: str) -> str:
    """Extract candidate name using SpaCy NER or fallback heuristics."""
    # 1. Try SpaCy NER first (looking for PERSON entities near the top)
    nlp = get_nlp()
    if nlp:
        # Only process the first 1000 characters to find the name quickly
        doc = nlp(text[:1000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Clean the name (remove newlines, extra spaces)
                clean_name = " ".join(ent.text.split()).title()
                # Basic validation: 2-4 words, no weird symbols
                if 2 <= len(clean_name.split()) <= 4 and re.match(r"^[A-Za-z.\-'\s]+$", clean_name):
                    return clean_name
                    
    # 2. Fallback heuristics if SpaCy fails
    ignore_words = {
        "resume", "curriculum", "vitae", "cv", "experience", "education",
        "skills", "profile", "summary", "objective", "projects", "certifications",
        "technologies", "languages", "personal", "details", "contact",
        "information", "technology", "specialist", "manager", "director",
        "developer", "engineer", "associate", "analyst", "administrator",
        "assistant", "lead", "senior", "junior", "operations", "support",
        "technician", "staff", "intern", "coordinator", "professional",
        "programmer", "consultant", "architect", "designer", "tester"
    }

    for line in text.split("\n")[:20]: # Only check top 20 lines
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if "@" in line or "http" in line or line.startswith("+") or any(char.isdigit() for char in line):
            continue
            
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.\-']+$", w) for w in words):
            line_lower_words = [w.lower() for w in words]
            if any(w in ignore_words for w in line_lower_words):
                continue
            return line.title()
            
    return "Unknown Candidate"


def _extract_email_from_resume(text: str) -> str:
    """Extract the first email address from the resume text."""
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else ""


# ── Candidates ─────────────────────────────────────────────────────────────


@router.get("/candidates")
async def get_candidates(skip: int = 0, limit: int = 50):
    """List all candidates with pagination."""
    candidates = await list_candidates(skip=skip, limit=limit)
    total = await count_candidates()
    return {"candidates": candidates, "total": total, "skip": skip, "limit": limit}


@router.get("/candidates/{candidate_id}")
async def get_candidate_detail(candidate_id: str):
    """Get a single candidate by ID."""
    candidate = await get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate


@router.delete("/candidates/{candidate_id}")
async def remove_candidate(candidate_id: str):
    """Delete a candidate from MongoDB and ChromaDB."""
    deleted = await delete_candidate(candidate_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Candidate not found")
    delete_from_index(candidate_id)
    return {"deleted": True, "candidate_id": candidate_id}


# ── Stats ──────────────────────────────────────────────────────────────────


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    chroma_stats = get_collection_stats()
    candidate_count = await count_candidates()
    return {
        "total_candidates_in_db": candidate_count,
        "vector_store": chroma_stats,
    }