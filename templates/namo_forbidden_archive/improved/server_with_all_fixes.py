"""Complete server implementation with security and performance fixes.

Place this file inside the `improved` folder. It adds the folder to
`sys.path` so local modules under `core/`, `utils/`, `security/` and
`memory_service_db_improved.py` can be imported when running directly.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import logging
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from uuid import uuid4

# Ensure the `improved` directory is on sys.path when running this file directly
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import local improved modules
try:
    from core.namo_omega_engine_improved import NaMoOmegaEngine
except Exception as e:
    NaMoOmegaEngine = None
    logging.getLogger(__name__).warning(f"Could not import engine: {e}")

try:
    from memory_service_db_improved import MemoryServiceDB
except Exception as e:
    MemoryServiceDB = None
    logging.getLogger(__name__).warning(f"Could not import memory service: {e}")

try:
    from utils.cache_improved import TTLCache
except Exception as e:
    TTLCache = None
    logging.getLogger(__name__).warning(f"Could not import TTLCache: {e}")

try:
    from security.auth_improved import setup_security
except Exception as e:
    setup_security = None
    logging.getLogger(__name__).warning(f"Could not import security helpers: {e}")

try:
    from core.session_manager_improved import ThreadSafeSessionManager
except Exception as e:
    ThreadSafeSessionManager = None
    logging.getLogger(__name__).warning(f"Could not import session manager: {e}")

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("namo.improved.server")

# ============ Configuration ============
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8081")
API_KEY = os.getenv("API_KEY", "default-key-change-in-production")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() in ("1", "true", "yes")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))


# ============ Models ============
class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, max_length=100)

    @validator("text")
    def text_not_empty(cls, v: str) -> str:  # type: ignore
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class MediaResponse(BaseModel):
    image: Optional[str] = None
    audio: Optional[str] = None
    tts: Optional[str] = None


class StatusResponse(BaseModel):
    arousal: Optional[str] = None
    sin_status: Optional[str] = None
    active_personas: list = []
    engine_health: Optional[str] = None
    response_time_ms: float = 0.0


class ChatResponse(BaseModel):
    response: str
    session_id: str
    media: MediaResponse
    status: StatusResponse
    request_id: str
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    engine: str
    sin: str
    memory_records: int
    active_sessions: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
    timestamp: datetime


class SessionStatsResponse(BaseModel):
    session_id: str
    created_at: str
    last_active: str
    message_count: int
    average_arousal: float
    arousal_history: list


# ============ Simple Rate Limiter & Auth ============
class SimpleRateLimiter:
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        self.logger = logging.getLogger("namo.improved.rate_limiter")

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        cutoff = now - 60
        if client_id not in self.requests:
            self.requests[client_id] = []
        self.requests[client_id] = [t for t in self.requests[client_id] if t > cutoff]
        if len(self.requests[client_id]) >= self.requests_per_minute:
            self.logger.warning(f"Rate limit exceeded for {client_id}")
            return False
        self.requests[client_id].append(now)
        return True


class SimpleAPIKeyAuth:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger("namo.improved.apikey")

    async def verify(self, x_api_key: Optional[str] = None) -> bool:
        if not x_api_key:
            return False
        if x_api_key != self.api_key:
            self.logger.warning(f"Invalid API key attempt: {x_api_key[:5]}***")
            return False
        return True


# ============ App Initialization ============
app = FastAPI(title="NaMo Omega Engine API v3.0 (improved)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Services initialization (safe)
engine = None
memory_service = None
response_cache = None

try:
    engine = NaMoOmegaEngine() if NaMoOmegaEngine else None
    logger.info("Loaded NaMoOmegaEngine")
except Exception as e:
    engine = None
    logger.exception(f"Engine init failed: {e}")

try:
    memory_service = MemoryServiceDB() if MemoryServiceDB else None
    logger.info("Loaded MemoryServiceDB")
except Exception as e:
    memory_service = None
    logger.exception(f"Memory service init failed: {e}")

try:
    if ENABLE_CACHING and TTLCache:
        response_cache = TTLCache(default_ttl=CACHE_TTL)
        logger.info("Initialized TTLCache")
    else:
        response_cache = None
except Exception as e:
    response_cache = None
    logger.exception(f"Cache init failed: {e}")

# Session manager fallback
session_manager = ThreadSafeSessionManager(session_timeout=SESSION_TIMEOUT_MINUTES * 60) if ThreadSafeSessionManager else None

# Simple components
rate_limiter = SimpleRateLimiter(requests_per_minute=100)
api_auth = SimpleAPIKeyAuth(api_key=API_KEY)
start_time = time.time()


# ============ Exception Handlers ============
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="ValidationError",
            detail=str(exc),
            request_id=getattr(request.state, "request_id", "unknown"),
            timestamp=datetime.utcnow(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred",
            request_id=getattr(request.state, "request_id", "unknown"),
            timestamp=datetime.utcnow(),
        ).dict(),
    )


# ============ Middleware ============
@app.middleware("http")
async def add_request_tracking(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    request.state.start_time = datetime.utcnow()

    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "TooManyRequests",
                "detail": "Rate limit exceeded. Max 100 requests per minute",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    logger.info({"request_id": request_id, "method": request.method, "path": request.url.path, "client": client_ip})

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"Request processing error: {e}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                detail=str(e),
                request_id=request_id,
                timestamp=datetime.utcnow(),
            ).dict(),
        )

    process_time = (datetime.utcnow() - request.state.start_time).total_seconds()
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)

    logger.info({"request_id": request_id, "status_code": response.status_code, "process_time_ms": process_time * 1000})
    return response


# Helper to support sync or async engine.process_input
async def _maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


# ============ Routes ============
@app.get("/", response_model=HealthResponse)
async def health_check():
    try:
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        memory_count = 0
        if memory_service:
            stats = memory_service.get_stats()
            memory_count = stats.get("total_memories", 0)

        uptime = time.time() - start_time
        active_sessions = engine.session_manager.get_session_count() if getattr(engine, "session_manager", None) else (session_manager.get_session_count() if session_manager else 0)

        return HealthResponse(
            status="NaMo is Horny & Online ✅",
            engine="Omega v3.0",
            sin="[Innocent Soul]",
            memory_records=memory_count,
            active_sessions=active_sessions,
            uptime_seconds=uptime,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request_model: ChatRequest, request: Request, x_api_key: Optional[str] = None):
    # Authenticate
    if not await api_auth.verify(x_api_key):
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")

    session_id = request_model.session_id or "default-session"
    logger.info(f"Processing chat for session: {session_id}")

    try:
        engine_result = engine.process_input(input_text=request_model.text, session_id=session_id, use_cache=True)
        engine_result = await _maybe_await(engine_result)

        # engine_result expected to have attributes: text, error, system_status
        if getattr(engine_result, "error", None):
            logger.warning(f"Engine error: {engine_result.error}")
            raise HTTPException(status_code=400, detail=str(engine_result.error))

        # optionally store memory
        if memory_service:
            try:
                memory_service.store_memory(
                    {
                        "content": f"user: {request_model.text}\nassistant: {getattr(engine_result, 'text', '')}",
                        "type": "contextual",
                        "session_id": session_id,
                        "emotion_context": None,
                        "dharma_tags": ["chat"],
                    }
                )
            except Exception:
                logger.exception("Failed to store memory")

        system_status = getattr(engine_result, "system_status", None) or {}

        status = StatusResponse(
            arousal=getattr(system_status, "arousal", None) if system_status else None,
            sin_status=getattr(system_status, "sin_status", None) if system_status else None,
            active_personas=getattr(system_status, "active_personas", []) if system_status else [],
            engine_health=getattr(system_status, "engine_health", None) if system_status else None,
            response_time_ms=float(getattr(system_status, "response_time_ms", 0.0) or 0.0),
        )

        return ChatResponse(
            response=str(getattr(engine_result, "text", "")),
            session_id=session_id,
            media=MediaResponse(),
            status=status,
            request_id=getattr(request.state, "request_id", "unknown"),
            timestamp=datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")


@app.get("/session/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(session_id: str, x_api_key: Optional[str] = None):
    if not await api_auth.verify(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")

    stats = engine.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionStatsResponse(**stats)


@app.post("/memory/store")
async def store_memory_endpoint(memory_data: Dict[str, Any], x_api_key: Optional[str] = None):
    if not await api_auth.verify(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not available")
    try:
        stored = memory_service.store_memory(memory_data)
        return stored
    except Exception as e:
        logger.exception(f"Store memory error: {e}")
        raise HTTPException(status_code=500, detail="Failed to store memory")


@app.post("/memory/recall")
async def recall_memories(query: str, limit: int = 10, x_api_key: Optional[str] = None):
    if not await api_auth.verify(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not available")
    if not query or len(query) < 1:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        memories = memory_service.recall_memories(query, limit)
        return {"count": len(memories), "memories": memories}
    except Exception as e:
        logger.exception(f"Recall error: {e}")
        raise HTTPException(status_code=500, detail="Failed to recall memories")


@app.get("/memory/stats")
async def memory_stats(x_api_key: Optional[str] = None):
    if not await api_auth.verify(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not available")
    try:
        return memory_service.get_stats()
    except Exception as e:
        logger.exception(f"Memory stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory stats")


@app.get("/debug/sessions")
async def debug_sessions(x_api_key: Optional[str] = None):
    if not await api_auth.verify(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    try:
        manager = getattr(engine, "session_manager", None) or session_manager
        return {
            "active_sessions": manager.get_session_count() if manager else 0,
            "sessions": manager.get_all_sessions() if manager and hasattr(manager, "get_all_sessions") else {},
        }
    except Exception as e:
        logger.exception(f"Debug sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sessions")


@app.post("/cleanup/memories")
async def cleanup_old_memories(days_old: int = 30, x_api_key: Optional[str] = None):
    if not await api_auth.verify(x_api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not available")
    if days_old < 1:
        raise HTTPException(status_code=400, detail="days_old must be >= 1")
    try:
        deleted = memory_service.cleanup_old_memories(days_old)
        return {"success": True, "deleted_count": deleted, "message": f"Deleted {deleted} memories older than {days_old} days"}
    except Exception as e:
        logger.exception(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup memories")


# ============ Startup / Shutdown ============
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("🚀 NaMo Omega Engine v3.0 (improved) Starting...")
    logger.info("=" * 60)
    if engine:
        logger.info("✅ Engine initialized")
    else:
        logger.error("❌ Engine failed to initialize")

    if memory_service:
        try:
            stats = memory_service.get_stats()
            logger.info(f"📊 Loaded {stats.get('total_memories', 0)} memories")
        except Exception:
            logger.exception("Failed to fetch memory stats on startup")
    else:
        logger.warning("⚠️ Memory service not available")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 NaMo Omega Engine shutting down...")
    manager = getattr(engine, "session_manager", None) or session_manager
    if manager and hasattr(manager, "cleanup_expired_sessions"):
        try:
            manager.cleanup_expired_sessions()
            logger.info("✅ Session cleanup complete")
        except Exception:
            logger.exception("Session cleanup failed")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
