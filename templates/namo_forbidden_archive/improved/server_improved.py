"""
Enhanced FastAPI REST server with better error handling and middleware.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import logging
import time
from datetime import datetime
import asyncio
from core.namo_omega_engine_improved import NaMoOmegaEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NaMo Omega Engine API",
    description="Enhanced Dark Dialogue Engine",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = NaMoOmegaEngine()


# ============ Request/Response Models ============

class ChatRequest(BaseModel):
    """Chat request model with validation."""
    text: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(None, max_length=100)
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


class MediaResponse(BaseModel):
    """Media URLs in response."""
    image: Optional[str] = None
    audio: Optional[str] = None
    tts: Optional[str] = None


class StatusResponse(BaseModel):
    """Status information."""
    arousal: str
    sin_status: str
    active_personas: list
    engine_health: str
    response_time_ms: float


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str
    media: MediaResponse = MediaResponse()
    status: StatusResponse
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine: str
    sin: str
    uptime_seconds: float


class SessionStatsResponse(BaseModel):
    """Session statistics."""
    session_id: str
    message_count: int
    average_arousal: float
    arousal_history: list


# ============ Middleware ============

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time to response headers."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(e)
            }
        )
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# ============ Error Handlers ============

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ============ Routes ============

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="NaMo is Horny & Online",
            engine="Omega v2.0",
            sin="[Innocent Soul] บาปสะสม: 0 | ปลดล็อก: ",
            uptime_seconds=time.time()
        )
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint."""
    try:
        logger.info(f"Processing chat request: {request.session_id}")
        
        # Process input through engine
        engine_response = engine.process_input(
            input_text=request.text,
            session_id=request.session_id or "default-session",
            use_cache=True
        )
        
        # Check for errors
        if engine_response.error:
            logger.warning(f"Engine error: {engine_response.error}")
            raise HTTPException(
                status_code=400,
                detail=engine_response.error
            )
        
        # Build response
        return ChatResponse(
            response=engine_response.text,
            session_id=request.session_id or "default-session",
            media=MediaResponse(),
            status=StatusResponse(
                arousal=engine_response.system_status.arousal if engine_response.system_status else "N/A",
                sin_status=engine_response.system_status.sin_status if engine_response.system_status else "N/A",
                active_personas=engine_response.system_status.active_personas if engine_response.system_status else [],
                engine_health=engine_response.system_status.engine_health if engine_response.system_status else "UNKNOWN",
                response_time_ms=engine_response.system_status.response_time_ms if engine_response.system_status else 0.0
            ),
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process chat request")


@app.get("/session/{session_id}/stats", response_model=Optional[SessionStatsResponse])
async def get_session_stats(session_id: str):
    """Get session statistics."""
    try:
        stats = engine.get_session_stats(session_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatsResponse(**stats)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session stats error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session stats")


@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to list active sessions (development only)."""
    return {
        "active_sessions": len(engine.session_manager.sessions),
        "sessions": [
            {
                "id": sid,
                "created_at": s["created_at"].isoformat(),
                "message_count": s["message_count"],
                "arousal_avg": sum(s.get("arousal_history", [0])) / max(1, len(s.get("arousal_history", [0]))),
            }
            for sid, s in list(engine.session_manager.sessions.items())[:10]
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
