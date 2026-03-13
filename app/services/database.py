"""MongoDB async operations using Motor."""

from __future__ import annotations

import logging
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level database references
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_db() -> None:
    """Connect to MongoDB."""
    global _client, _db
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db_name]

    # Verify connection
    await _client.admin.command("ping")
    logger.info(f"Connected to MongoDB: {settings.mongodb_db_name}")


async def close_db() -> None:
    """Close the MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    """Return the database instance."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call connect_db() first.")
    return _db


# ── Candidates ───────────────────────────────────────────────────────────────


async def insert_candidate(data: dict) -> str:
    """Insert a new candidate document. Returns the string ID."""
    db = get_db()
    result = await db.candidates.insert_one(data)
    return str(result.inserted_id)


async def get_candidate(candidate_id: str) -> Optional[dict]:
    """Fetch a candidate by ID. Returns None if not found."""
    db = get_db()
    try:
        doc = await db.candidates.find_one({"_id": ObjectId(candidate_id)})
    except Exception:
        doc = await db.candidates.find_one({"_id": candidate_id})

    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def find_candidate_by_email(email: str) -> Optional[dict]:
    """Find a candidate by email. Returns None if not found."""
    if not email:
        return None
    db = get_db()
    doc = await db.candidates.find_one({"email": email})
    if doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


async def delete_candidate(candidate_id: str) -> bool:
    """Delete a candidate by ID. Returns True if deleted."""
    db = get_db()
    try:
        result = await db.candidates.delete_one({"_id": ObjectId(candidate_id)})
    except Exception:
        result = await db.candidates.delete_one({"_id": candidate_id})
    deleted = result.deleted_count > 0
    if deleted:
        logger.info(f"Deleted candidate: {candidate_id}")
    return deleted


async def list_candidates(skip: int = 0, limit: int = 50) -> list:
    """List candidates with pagination."""
    db = get_db()
    cursor = db.candidates.find().skip(skip).limit(limit)
    candidates = await cursor.to_list(length=limit)
    for c in candidates:
        c["id"] = str(c.pop("_id"))
    return candidates


async def count_candidates() -> int:
    """Return the total number of candidates."""
    db = get_db()
    return await db.candidates.count_documents({})


# ── Screenings ───────────────────────────────────────────────────────────────


async def save_screening(record: dict) -> str:
    """Persist a screening result. Returns the string ID."""
    db = get_db()
    result = await db.screenings.insert_one(record)
    logger.info(f"Saved screening record: {result.inserted_id}")
    return str(result.inserted_id)
