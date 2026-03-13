from __future__ import annotations

import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    """Load the MPNet embedding model (singleton)."""
    global _model
    if _model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_chroma_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (singleton)."""
    global _chroma_client, _collection
    if _collection is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{settings.chroma_collection}' ready "
            f"({_collection.count()} documents)"
        )
    return _collection


def embed_text(text: str) -> list[float]:
    """Generate an MPNet embedding for a single text."""
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def index_resume(candidate_id: str, resume_text: str, metadata: dict) -> str:
    """
    Index a single resume into ChromaDB.
    Returns the document ID.
    """
    collection = get_chroma_collection()
    embedding = embed_text(resume_text)

    collection.upsert(
        ids=[candidate_id],
        embeddings=[embedding],
        documents=[resume_text],
        metadatas=[metadata],
    )
    logger.info(f"Indexed resume for candidate: {candidate_id}")
    return candidate_id


def index_resumes_batch(
    candidate_ids: list[str],
    resume_texts: list[str],
    metadatas: list[dict],
) -> int:
    """Index multiple resumes in a single batch. Returns count indexed."""
    collection = get_chroma_collection()
    embeddings = embed_batch(resume_texts)

    collection.upsert(
        ids=candidate_ids,
        embeddings=embeddings,
        documents=resume_texts,
        metadatas=metadatas,
    )
    logger.info(f"Batch indexed {len(candidate_ids)} resumes")
    return len(candidate_ids)


def search_similar_candidates(
    job_description: str,
    top_k: int = 10,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Query ChromaDB for the most similar resumes to a job description.
    Returns list of {id, document, metadata, distance} dicts.
    """
    collection = get_chroma_collection()

    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty, no candidates to search")
        return []

    query_embedding = embed_text(job_description)

    total_docs = collection.count()
    safe_top_k = min(top_k, total_docs)
    
    candidates = []

    if safe_top_k > 20 and total_docs <= 200:
        # ChromaDB HNSW crashes if n_results is too close to total collection size
        # For small local datasets, manual cosine similarity fallback is perfectly fast.
        all_docs = collection.get(include=["documents", "metadatas", "embeddings"])
        if all_docs and all_docs.get("ids"):
            for i in range(len(all_docs["ids"])):
                doc_emb = all_docs["embeddings"][i]
                
                # Math.sqrt equivalent
                dot = sum(a * b for a, b in zip(query_embedding, doc_emb))
                norm1 = sum(a * a for a in query_embedding) ** 0.5
                norm2 = sum(a * a for a in doc_emb) ** 0.5
                sim = dot / (norm1 * norm2) if norm1 and norm2 else 0.0
                
                if sim >= threshold:
                    candidates.append({
                        "id": all_docs["ids"][i],
                        "document": all_docs["documents"][i],
                        "metadata": all_docs["metadatas"][i],
                        "similarity": round(sim, 4),
                    })
            
            # Sort by similarity desc and trim
            candidates.sort(key=lambda x: x["similarity"], reverse=True)
            candidates = candidates[:safe_top_k]
    else:
        # Standard fast HNSW vector search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k,
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(results["ids"][0])):
            similarity = 1.0 - results["distances"][0][i]
            if similarity >= threshold:
                candidates.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": round(similarity, 4),
                })

    logger.info(
        f"Found {len(candidates)} candidates above threshold {threshold} "
        f"(searched {total_docs} total)"
    )
    return candidates


def delete_from_index(candidate_id: str) -> bool:
    """Remove a candidate's embedding from ChromaDB."""
    collection = get_chroma_collection()
    try:
        collection.delete(ids=[candidate_id])
        logger.info(f"Deleted from index: {candidate_id}")
        return True
    except Exception as e:
        logger.warning(f"Could not delete {candidate_id} from index: {e}")
        return False


def get_collection_stats() -> dict:
    """Return basic stats about the ChromaDB collection."""
    collection = get_chroma_collection()
    return {
        "collection_name": collection.name,
        "total_documents": collection.count(),
    }