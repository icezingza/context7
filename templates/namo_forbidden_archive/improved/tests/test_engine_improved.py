"""
Comprehensive test suite for improved engine.
"""
import pytest
from datetime import datetime
from core.namo_omega_engine_improved import (
    NaMoOmegaEngine,
    ArousalDetectionMatrix,
    IntensityLevel,
    SessionManager,
    DialogueBank,
    ArousalState
)


class TestArousalDetection:
    """Test arousal detection."""
    
    def test_arousal_level_bounds(self):
        """Test arousal level stays within bounds."""
        detector = ArousalDetectionMatrix()
        result = detector.detect_arousal("test input")
        
        assert 0.0 <= result.level <= 1.0
    
    def test_arousal_categorization(self):
        """Test arousal intensity categorization."""
        detector = ArousalDetectionMatrix()
        
        # Test low intensity
        low_result = detector.detect_arousal("hello")
        assert low_result.category in [IntensityLevel.LOW, IntensityLevel.MEDIUM]
    
    def test_caching(self):
        """Test detection caching."""
        detector = ArousalDetectionMatrix()
        
        result1 = detector.detect_arousal("test", use_cache=True)
        result2 = detector.detect_arousal("test", use_cache=True)
        
        assert result1.level == result2.level


class TestSessionManager:
    """Test session management."""
    
    def test_session_creation(self):
        """Test session creation."""
        manager = SessionManager()
        session = manager.create_session("test-session")
        
        assert session["created_at"] is not None
        assert session["message_count"] == 0
    
    def test_session_retrieval(self):
        """Test session retrieval."""
        manager = SessionManager()
        manager.create_session("test-session")
        
        retrieved = manager.get_session("test-session")
        assert retrieved is not None
    
    def test_session_expiration(self):
        """Test session expiration logic."""
        manager = SessionManager(session_timeout_minutes=1)
        session = manager.create_session("test-session")
        
        # Session should not be expired immediately
        assert not manager._is_expired(session)


class TestNaMoEngine:
    """Test NaMo Omega Engine."""
    
    def test_process_input_valid(self):
        """Test valid input processing."""
        engine = NaMoOmegaEngine()
        response = engine.process_input("สวัสดี", "test-session")
        
        assert response.text != ""
        assert response.error is None
    
    def test_process_input_empty(self):
        """Test empty input handling."""
        engine = NaMoOmegaEngine()
        response = engine.process_input("", "test-session")
        
        assert response.error is not None
    
    def test_process_input_too_long(self):
        """Test long input handling."""
        engine = NaMoOmegaEngine()
        long_text = "a" * 6000
        response = engine.process_input(long_text, "test-session")
        
        assert response.error is not None
    
    def test_session_stats(self):
        """Test session statistics."""
        engine = NaMoOmegaEngine()
        engine.process_input("สวัสดี", "test-session")
        
        stats = engine.get_session_stats("test-session")
        assert stats is not None
        assert stats["message_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
