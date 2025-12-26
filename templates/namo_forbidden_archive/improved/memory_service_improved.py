"""
Enhanced Memory Service with better indexing and retrieval.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import json
from pathlib import Path
import asyncio
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NaMo Memory Service", version="2.0.0")


# ============ Models ============

class EmotionContext(BaseModel):
    """Emotion context model."""
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    emotion_type: str
    intensity: int = Field(..., ge=1, le=10)


class MemoryRecord(BaseModel):
    """Memory record model."""
    content: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    emotion_context: Optional[EmotionContext] = None
    dharma_tags: Optional[List[str]] = None
    id: Optional[str] = None
    created_at: Optional[datetime] = None


class MemoryStoreRequest(BaseModel):
    """Request to store memory."""
    content: str
    type: str
    session_id: str
    emotion_context: Optional[EmotionContext] = None
    dharma_tags: Optional[List[str]] = None


class MemoryRecallRequest(BaseModel):
    """Request to recall memory."""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    memory_records: int
    storage_used_mb: float


class MemoryService:
    """In-memory storage service for memories."""
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.memories: Dict[str, MemoryRecord] = {}
        self.session_index: Dict[str, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.persistence_path = Path(persistence_path or "memories.json")
        self.logger = logging.getLogger(f"{__name__}.MemoryService")
        self._load_from_disk()
    
    def store_memory(self, memory: MemoryRecord) -> MemoryRecord:
        """Store a memory with automatic indexing."""
        try:
            # Generate ID if not provided
            if not memory.id:
                memory.id = f"mem_{int(datetime.now().timestamp())}_{len(self.memories)}"
            
            # Set creation timestamp
            if not memory.created_at:
                memory.created_at = datetime.now()
            
            # Store memory
            self.memories[memory.id] = memory
            
            # Update indexes
            self.session_index[memory.session_id].append(memory.id)
            
            if memory.dharma_tags:
                for tag in memory.dharma_tags:
                    self.tag_index[tag].append(memory.id)
            
            self.logger.info(f"Stored memory: {memory.id}")
            
            # Persist to disk
            asyncio.create_task(self._persist_to_disk())
            
            return memory
        
        except Exception as e:
            self.logger.error(f"Error storing memory: {str(e)}")
            raise
    
    def recall_memories(self, query: str, limit: int = 10) -> List[MemoryRecord]:
        """Recall memories matching query."""
        try:
            results = []
            
            # Search in content and session IDs
            query_lower = query.lower()
            
            for memory in self.memories.values():
                # Check if query matches content or session
                if (query_lower in memory.content.lower() or
                    query_lower in memory.session_id.lower() or
                    (memory.dharma_tags and any(query_lower in tag.lower() for tag in memory.dharma_tags))):
                    
                    results.append(memory)
            
            # Sort by creation date (newest first)
            results.sort(key=lambda x: x.created_at or datetime.now(), reverse=True)
            
            # Return limited results
            return results[:limit]
        
        except Exception as e:
            self.logger.error(f"Error recalling memories: {str(e)}")
            return []
    
    def get_session_memories(self, session_id: str) -> List[MemoryRecord]:
        """Get all memories for a session."""
        try:
            memory_ids = self.session_index.get(session_id, [])
            return [self.memories[mid] for mid in memory_ids if mid in self.memories]
        
        except Exception as e:
            self.logger.error(f"Error getting session memories: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        total_size = sum(
            len(json.dumps(m.dict()).encode('utf-8'))
            for m in self.memories.values()
        )
        
        return {
            "total_memories": len(self.memories),
            "total_sessions": len(self.session_index),
            "total_tags": len(self.tag_index),
            "storage_used_bytes": total_size,
            "storage_used_mb": total_size / (1024 * 1024)
        }
    
    async def _persist_to_disk(self):
        """Persist memories to disk."""
        try:
            data = {
                mid: m.dict()
                for mid, m in self.memories.items()
            }
            
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str, indent=2)
            
            self.logger.debug("Memories persisted to disk")
        
        except Exception as e:
            self.logger.error(f"Error persisting to disk: {str(e)}")
    
    def _load_from_disk(self):
        """Load memories from disk."""
        try:
            if self.persistence_path.exists():
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for mid, mdata in data.items():
                    # Parse memory record
                    mdata['created_at'] = datetime.fromisoformat(mdata['created_at'])
                    
                    memory = MemoryRecord(**mdata)
                    self.memories[mid] = memory
                    
                    # Rebuild indexes
                    self.session_index[memory.session_id].append(mid)
                    if memory.dharma_tags:
                        for tag in memory.dharma_tags:
                            self.tag_index[tag].append(mid)
                
                self.logger.info(f"Loaded {len(self.memories)} memories from disk")
        
        except Exception as e:
            self.logger.warning(f"Error loading memories from disk: {str(e)}")


# Initialize memory service
memory_service = MemoryService()


# ============ Routes ============

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    stats = memory_service.get_stats()
    return HealthResponse(
        status="ok",
        memory_records=stats["total_memories"],
        storage_used_mb=stats["storage_used_mb"]
    )


@app.post("/store", response_model=MemoryRecord)
async def store(request: MemoryStoreRequest):
    """Store a memory."""
    try:
        memory = MemoryRecord(**request.dict())
        return memory_service.store_memory(memory)
    
    except Exception as e:
        logger.error(f"Store error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to store memory")


@app.post("/recall", response_model=List[MemoryRecord])
async def recall(request: MemoryRecallRequest):
    """Recall memories by query."""
    try:
        return memory_service.recall_memories(request.query, request.limit)
    
    except Exception as e:
        logger.error(f"Recall error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to recall memories")


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get all memories for a session."""
    try:
        memories = memory_service.get_session_memories(session_id)
        return {"session_id": session_id, "memories": memories}
    
    except Exception as e:
        logger.error(f"Get session error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return memory_service.get_stats()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8081,
        log_level="info"
    )
