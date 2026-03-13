import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.api.routes import router
from app.services.database import connect_db, close_db

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("Starting Agentic Hiring Screener")
    await connect_db()
    yield
    await close_db()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Agentic Hiring Screener",
    description=(
        "Multi-agent candidate screening pipeline powered by "
        "LangGraph, Mistral, ChromaDB (MPNet embeddings), and MongoDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["screening"])

# Serve static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agentic-hiring-screener"}