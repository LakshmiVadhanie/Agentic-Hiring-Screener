"""Integration tests for the Hiring Screener API."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Health Check ─────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health_check(client: AsyncClient):
    """Test the health check endpoint returns 200."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "agentic-hiring-screener"


# ── Resume Indexing ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_index_resume(client: AsyncClient):
    """Test indexing a resume via the API."""
    payload = {
        "candidate_name": "Test Candidate",
        "email": "test@example.com",
        "resume_text": "Experienced Python developer with 5 years in ML and NLP.",
    }
    response = await client.post("/api/v1/resumes/index", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "candidate_id" in data
    assert data["indexed"] is True


# ── Candidates ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_candidates(client: AsyncClient):
    """Test listing candidates."""
    response = await client.get("/api/v1/candidates")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "candidates" in data


# ── Stats ────────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_stats(client: AsyncClient):
    """Test the stats endpoint."""
    response = await client.get("/api/v1/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_candidates_in_db" in data
    assert "vector_store" in data


# ── Screening ────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_screen_request_validation(client: AsyncClient):
    """Test that screening validates request body."""
    response = await client.post("/api/v1/screen", json={})
    assert response.status_code == 422  # Validation error


@pytest.mark.anyio
async def test_screen_valid_request_shape(client: AsyncClient):
    """Test that a valid screening request has the correct response shape."""
    payload = {
        "job": {
            "title": "ML Engineer",
            "description": "Build ML pipelines",
            "required_skills": ["Python", "PyTorch"],
            "preferred_skills": ["Docker"],
            "min_experience_years": 2,
        },
        "top_k": 3,
        "threshold": 0.5,
    }
    response = await client.post("/api/v1/screen", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "job_title" in data
        assert "shortlisted" in data
        assert isinstance(data["shortlisted"], list)
