"""
API Tests for El-Harezmi Router

Tests for /api/v2/* endpoints.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import modules to test
from config.feature_flags import (
    FeatureFlags, PipelineVersion,
    get_feature_flags, set_feature_flags,
    get_pipeline_version, set_rollout_percentage
)


class TestFeatureFlags:
    """Test feature flag system"""
    
    def setup_method(self):
        """Reset feature flags before each test"""
        set_feature_flags(FeatureFlags())
    
    def test_default_is_legacy(self):
        """Default pipeline should be legacy"""
        flags = FeatureFlags()
        version = flags.get_pipeline_for_request()
        assert version == PipelineVersion.LEGACY
    
    def test_full_rollout_is_el_harezmi(self):
        """100% rollout should always return El-Harezmi"""
        flags = FeatureFlags(el_harezmi_rollout=1.0)
        
        # Test 100 times to ensure consistency
        for _ in range(100):
            version = flags.get_pipeline_for_request()
            assert version == PipelineVersion.EL_HAREZMI
    
    def test_force_override(self):
        """Force override should ignore rollout"""
        flags = FeatureFlags(
            el_harezmi_rollout=0.0,  # No rollout
            force_pipeline="el_harezmi"  # But force El-Harezmi
        )
        version = flags.get_pipeline_for_request()
        assert version == PipelineVersion.EL_HAREZMI
    
    def test_request_force_override(self):
        """Request-level force should override all"""
        flags = FeatureFlags(
            el_harezmi_rollout=0.0,
            force_pipeline="legacy"
        )
        # Request wants El-Harezmi
        version = flags.get_pipeline_for_request(force_version="el_harezmi")
        assert version == PipelineVersion.EL_HAREZMI
    
    def test_ab_test_consistent_routing(self):
        """A/B test should route same user consistently"""
        flags = FeatureFlags(
            el_harezmi_rollout=0.0,
            ab_test_enabled=True,
            ab_test_group_a="legacy",
            ab_test_group_b="el_harezmi"
        )
        
        user_id = "test_user_123"
        first_version = flags.get_pipeline_for_request(user_id=user_id)
        
        # Same user should always get same version
        for _ in range(10):
            version = flags.get_pipeline_for_request(user_id=user_id)
            assert version == first_version
    
    def test_rollout_percentage(self):
        """Rollout percentage should affect distribution"""
        flags = FeatureFlags(el_harezmi_rollout=0.5)
        
        el_harezmi_count = 0
        total = 1000
        
        for _ in range(total):
            version = flags.get_pipeline_for_request()
            if version == PipelineVersion.EL_HAREZMI:
                el_harezmi_count += 1
        
        # Should be roughly 50% (with some tolerance)
        percentage = el_harezmi_count / total
        assert 0.4 < percentage < 0.6, f"Expected ~50%, got {percentage*100:.1f}%"
    
    def test_set_rollout_percentage(self):
        """set_rollout_percentage should update flags"""
        set_rollout_percentage(0.75)
        flags = get_feature_flags()
        assert flags.el_harezmi_rollout == 0.75
    
    def test_rollout_clamp(self):
        """Rollout percentage should be clamped to 0-1"""
        set_rollout_percentage(1.5)
        flags = get_feature_flags()
        assert flags.el_harezmi_rollout == 1.0
        
        set_rollout_percentage(-0.5)
        flags = get_feature_flags()
        assert flags.el_harezmi_rollout == 0.0


class TestChatRequest:
    """Test ChatRequest model validation"""
    
    def test_valid_request(self):
        from src.api.el_harezmi_router import ChatRequest
        
        request = ChatRequest(
            message="EABC-3000 tork ayarı nasıl yapılır?",
            product_model="EABC-3000",
            language="tr"
        )
        
        assert request.message == "EABC-3000 tork ayarı nasıl yapılır?"
        assert request.product_model == "EABC-3000"
        assert request.language == "tr"
    
    def test_minimal_request(self):
        from src.api.el_harezmi_router import ChatRequest
        
        request = ChatRequest(message="Merhaba")
        
        assert request.message == "Merhaba"
        assert request.product_model is None
        assert request.language == "tr"  # Default
    
    def test_english_language(self):
        from src.api.el_harezmi_router import ChatRequest
        
        request = ChatRequest(
            message="How to configure EABC-3000?",
            language="en"
        )
        
        assert request.language == "en"
    
    def test_force_pipeline(self):
        from src.api.el_harezmi_router import ChatRequest
        
        request = ChatRequest(
            message="Test query",
            force_pipeline="el_harezmi"
        )
        
        assert request.force_pipeline == "el_harezmi"


class TestChatResponse:
    """Test ChatResponse model"""
    
    def test_response_fields(self):
        from src.api.el_harezmi_router import ChatResponse, SourceInfo
        
        response = ChatResponse(
            response="Test response",
            intent="configuration",
            confidence=0.95,
            product_model="EABC-3000",
            sources=[
                SourceInfo(document="test.pdf", score=0.8)
            ],
            pipeline_version="el_harezmi",
            language="tr",
            validation_status="allow",
            processing_time_ms=150.5,
            warnings=["Test warning"]
        )
        
        assert response.response == "Test response"
        assert response.intent == "configuration"
        assert response.confidence == 0.95
        assert len(response.sources) == 1
        assert response.sources[0].document == "test.pdf"


class TestHealthEndpoint:
    """Test health endpoint"""
    
    @pytest.mark.asyncio
    async def test_health_response(self):
        from src.api.el_harezmi_router import health_check
        
        # Reset flags
        set_feature_flags(FeatureFlags(el_harezmi_rollout=0.5))
        
        response = await health_check()
        
        assert response.status == "healthy"
        assert response.pipeline_version == "el_harezmi_v1"
        assert response.rollout_percentage == 0.5
        assert response.timestamp is not None


class TestPipelineSelection:
    """Test pipeline selection logic"""
    
    def test_legacy_pipeline_selected(self):
        """With 0% rollout, legacy should be selected"""
        set_feature_flags(FeatureFlags(el_harezmi_rollout=0.0))
        
        version = get_pipeline_version()
        assert version == PipelineVersion.LEGACY
    
    def test_el_harezmi_pipeline_selected(self):
        """With 100% rollout, el_harezmi should be selected"""
        set_feature_flags(FeatureFlags(el_harezmi_rollout=1.0))
        
        version = get_pipeline_version()
        assert version == PipelineVersion.EL_HAREZMI
    
    def test_v2_alias(self):
        """v2 should be alias for el_harezmi"""
        set_feature_flags(FeatureFlags())
        
        version = get_pipeline_version(force="v2")
        assert version == PipelineVersion.EL_HAREZMI


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
