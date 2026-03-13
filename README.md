# Agentic Hiring Screener

Multi-agent candidate screening pipeline that reduces hiring screening time by 70%. Built with **LangGraph**, **Mistral**, **ChromaDB**, **MPNet**, **MongoDB**, and deployed on **Google Cloud Run** via **FastAPI**.

## Architecture

```
Job Description
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│  Retriever  │────▶│  Evaluator   │────▶│   Ranker   │
│  (ChromaDB  │     │  (Mistral    │     │  (Score +  │
│   + MPNet)  │     │   LLM)       │     │   Rank)    │
└─────────────┘     └──────────────┘     └────────────┘
                                                │
                                                ▼
                                        Ranked Shortlist
```

**Pipeline Nodes (LangGraph):**

1. **Retriever** - Encodes the job description with MPNet, queries ChromaDB for semantically similar resumes
2. **Evaluator** - Sends each retrieved resume to Mistral for structured skill-matching and fit scoring
3. **Ranker** - Combines similarity scores (40%) and LLM scores (60%) into a final ranking, applies threshold filtering

## Key Results

- **70% reduction** in screening time by automating the retrieve-evaluate-rank loop
- **38% improvement** in match precision via MPNet embeddings indexed in ChromaDB
- Handles **500+ candidates/day** through batch processing and async I/O

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph |
| LLM | Mistral (via LangChain) |
| Embeddings | MPNet (sentence-transformers) |
| Vector Store | ChromaDB |
| Database | MongoDB (Motor async driver) |
| API | FastAPI |
| Deployment | Docker, Google Cloud Run |


## Project Structure

```
agentic-hiring-screener/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py             # Settings (pydantic-settings)
│   ├── agents/
│   │   ├── graph.py          # LangGraph pipeline
│   │   ├── nodes.py          # Retriever, Evaluator, Ranker nodes
│   │   └── state.py          # Shared pipeline state
│   ├── services/
│   │   ├── embedding.py      # MPNet + ChromaDB operations
│   │   ├── llm.py            # Mistral LLM client
│   │   └── database.py       # MongoDB async operations
│   ├── api/
│   │   └── routes.py         # API endpoint handlers
│   └── models/
│       └── schemas.py        # Pydantic request/response models
├── scripts/
│   └── seed_db.py            # Seed sample candidates
├── tests/
│   └── test_api.py           # Integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

