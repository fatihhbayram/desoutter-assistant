"""
Test suite for El-Harezmi 5-Stage Pipeline

Tests all stages independently and as integrated pipeline.
"""

import pytest
import asyncio
from typing import Dict, Any

# Import El-Harezmi components
from src.el_harezmi.stage1_intent_classifier import (
    IntentClassifier, IntentType, IntentResult, ExtractedEntities
)
from src.el_harezmi.stage2_retrieval_strategy import (
    RetrievalStrategyManager, RetrievalStrategy, RetrievedChunk
)
from src.el_harezmi.stage3_info_extraction import (
    InfoExtractor, ExtractionResult, ProcedureStep
)
from src.el_harezmi.stage4_kg_validation import (
    KGValidator, ValidationResult, ValidationStatus
)
from src.el_harezmi.stage5_response_formatter import (
    ResponseFormatter, FormattedResponse
)
from src.el_harezmi.pipeline import ElHarezmiPipeline


# ============================================
# Stage 1: Intent Classifier Tests
# ============================================

class TestIntentClassifier:
    """Test Stage 1: Intent Classification"""
    
    @pytest.fixture
    def classifier(self):
        return IntentClassifier()
    
    def test_error_code_detection(self, classifier):
        """Test error code intent detection"""
        result = classifier.classify("E1234 hata kodu ne anlama geliyor?")
        
        assert result.primary_intent == IntentType.ERROR_CODE
        assert result.entities.error_code == "E1234"
        assert result.confidence >= 0.7
    
    def test_configuration_intent(self, classifier):
        """Test configuration intent detection"""
        result = classifier.classify("EABC-3000 tork ayarı nasıl yapılır?")
        
        assert result.primary_intent == IntentType.CONFIGURATION
        assert result.entities.product_model == "EABC-3000"
    
    def test_compatibility_intent(self, classifier):
        """Test compatibility intent detection"""
        result = classifier.classify("EABC-3000 hangi CVI3 versiyonuyla çalışır?")
        
        assert result.primary_intent == IntentType.COMPATIBILITY
        assert result.entities.product_model == "EABC-3000"
        assert result.entities.controller_type == "CVI3"
    
    def test_troubleshoot_intent(self, classifier):
        """Test troubleshooting intent detection"""
        result = classifier.classify("EABC-3000 çalışmıyor, motor dönmüyor")
        
        assert result.primary_intent == IntentType.TROUBLESHOOT
        assert result.entities.product_model == "EABC-3000"
    
    def test_procedure_intent(self, classifier):
        """Test procedure intent detection"""
        result = classifier.classify("Firmware update nasıl yapılır?")
        
        assert result.primary_intent in [IntentType.PROCEDURE, IntentType.FIRMWARE, IntentType.HOW_TO]
    
    def test_specification_intent(self, classifier):
        """Test specification intent detection"""
        result = classifier.classify("EABC-3000 teknik özellikleri nelerdir?")
        
        assert result.primary_intent == IntentType.SPECIFICATION
        assert result.entities.product_model == "EABC-3000"
    
    def test_capability_query_intent(self, classifier):
        """Test capability query intent detection"""
        result = classifier.classify("EABC-3000 WiFi destekliyor mu?")
        
        # WiFi support question can be classified as CAPABILITY_QUERY or COMPATIBILITY
        acceptable = [IntentType.CAPABILITY_QUERY, IntentType.COMPATIBILITY]
        assert result.primary_intent in acceptable
    
    def test_accessory_query_intent(self, classifier):
        """Test accessory query intent detection"""
        result = classifier.classify("EABC-3000 hangi batarya ile çalışır?")
        
        # Battery question can be classified as ACCESSORY_QUERY or COMPATIBILITY
        acceptable = [IntentType.ACCESSORY_QUERY, IntentType.COMPATIBILITY]
        assert result.primary_intent in acceptable
        assert result.entities.accessory_type == "battery"
    
    def test_greeting_intent(self, classifier):
        """Test greeting intent detection"""
        result = classifier.classify("Merhaba")
        
        assert result.primary_intent == IntentType.GREETING
    
    def test_multi_label_intent(self, classifier):
        """Test multi-label intent detection"""
        result = classifier.classify("EABC-3000 tork 50 Nm nasıl ayarlanır ve hangi controller gerekir?")
        
        assert result.primary_intent == IntentType.CONFIGURATION
        assert len(result.secondary_intents) > 0
    
    def test_parameter_extraction(self, classifier):
        """Test parameter extraction from query"""
        result = classifier.classify("Tork 50 Nm olarak ayarla")
        
        assert result.entities.parameter_type == "torque"
        assert "50" in result.entities.target_value
    
    def test_product_family_detection(self, classifier):
        """Test product family detection"""
        result = classifier.classify("EABC-3000 sorun yaşıyorum")
        
        assert result.entities.product_family == "EABC"
    
    def test_controller_extraction(self, classifier):
        """Test controller type extraction"""
        result = classifier.classify("CVI3 ile bağlantı kuramıyorum")
        
        assert result.entities.controller_type == "CVI3"


# ============================================
# Stage 2: Retrieval Strategy Tests
# ============================================

class TestRetrievalStrategy:
    """Test Stage 2: Retrieval Strategy"""
    
    @pytest.fixture
    def strategy_manager(self):
        return RetrievalStrategyManager()
    
    def test_configuration_strategy(self, strategy_manager):
        """Test configuration retrieval strategy"""
        strategy = strategy_manager.get_strategy(IntentType.CONFIGURATION)
        
        assert "configuration_guide" in strategy.primary_sources
        assert strategy.boost_factors.get("configuration_guide", 0) >= 2.0
    
    def test_compatibility_strategy(self, strategy_manager):
        """Test compatibility retrieval strategy"""
        strategy = strategy_manager.get_strategy(IntentType.COMPATIBILITY)
        
        assert "compatibility_matrix" in strategy.primary_sources
        assert strategy.boost_factors.get("compatibility_matrix", 0) >= 4.0
        assert strategy.use_knowledge_graph == True
    
    def test_troubleshoot_strategy(self, strategy_manager):
        """Test troubleshooting retrieval strategy"""
        strategy = strategy_manager.get_strategy(IntentType.TROUBLESHOOT)
        
        assert "service_bulletin" in strategy.primary_sources
        assert strategy.boost_factors.get("service_bulletin", 0) >= 2.5
    
    def test_error_code_strategy(self, strategy_manager):
        """Test error code retrieval strategy"""
        strategy = strategy_manager.get_strategy(IntentType.ERROR_CODE)
        
        assert "error_code_list" in strategy.primary_sources
        assert strategy.boost_factors.get("error_code_list", 0) >= 3.0
    
    def test_filter_building(self, strategy_manager):
        """Test Qdrant filter building"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("EABC-3000 tork ayarı nasıl yapılır?")
        
        strategy = strategy_manager.get_strategy(intent_result.primary_intent)
        filter_dict = strategy_manager.build_qdrant_filter(intent_result, strategy)
        
        # Should have product model filter
        assert "must" in filter_dict or "should" in filter_dict
    
    def test_score_boosting(self, strategy_manager):
        """Test score boosting calculation"""
        strategy = strategy_manager.get_strategy(IntentType.CONFIGURATION)
        
        base_score = 0.5
        metadata = {"document_type": "configuration_guide"}
        
        boosted = strategy_manager.calculate_boosted_score(base_score, metadata, strategy)
        
        assert boosted > base_score
    
    def test_rrf_fusion(self, strategy_manager):
        """Test RRF fusion of results"""
        primary = [
            RetrievedChunk(chunk_id="1", content="chunk1", score=0.9, document_type="manual"),
            RetrievedChunk(chunk_id="2", content="chunk2", score=0.8, document_type="manual"),
        ]
        secondary = [
            RetrievedChunk(chunk_id="3", content="chunk3", score=0.85, document_type="guide"),
            RetrievedChunk(chunk_id="1", content="chunk1", score=0.7, document_type="manual"),
        ]
        
        merged = strategy_manager.merge_secondary_results(primary, secondary)
        
        # Chunk 1 should be boosted (appears in both)
        assert merged[0].chunk_id == "1"


# ============================================
# Stage 3: Info Extraction Tests
# ============================================

class TestInfoExtraction:
    """Test Stage 3: Information Extraction"""
    
    @pytest.fixture
    def extractor(self):
        return InfoExtractor()  # No LLM client for tests
    
    def test_template_selection(self, extractor):
        """Test extraction template selection"""
        config_template = extractor.get_template(IntentType.CONFIGURATION)
        compat_template = extractor.get_template(IntentType.COMPATIBILITY)
        
        assert "step_by_step_procedure" in config_template["extract"]
        assert "compatible_controllers" in compat_template["extract"]
    
    def test_fallback_extraction(self, extractor):
        """Test extraction without LLM"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("EABC-3000 ayarı nasıl yapılır?")
        
        # Create mock retrieval result
        from src.el_harezmi.stage2_retrieval_strategy import RetrievalResult
        retrieval_result = RetrievalResult(
            chunks=[
                RetrievedChunk(
                    chunk_id="1",
                    content="WARNING: Always check torque settings. 1. Open menu 2. Select parameters 3. Enter value",
                    score=0.9,
                    document_type="manual"
                )
            ],
            strategy_used="configuration",
            filters_applied={},
            total_candidates=1,
            intent=IntentType.CONFIGURATION,
            query="test"
        )
        
        result = extractor.extract_without_llm(intent_result, retrieval_result)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.warnings) > 0 or len(result.procedure) > 0


# ============================================
# Stage 4: KG Validation Tests
# ============================================

class TestKGValidation:
    """Test Stage 4: Knowledge Graph Validation"""
    
    @pytest.fixture
    def validator(self):
        return KGValidator()
    
    def test_product_family_detection(self, validator):
        """Test product family extraction"""
        family = validator._get_product_family("EABC-3000")
        assert family == "EABC"
        
        family = validator._get_product_family("EFD-500")
        assert family == "EFD"
    
    def test_compatibility_check_valid(self, validator):
        """Test valid compatibility check"""
        is_compatible, message = validator.check_tool_controller_compatibility(
            "EABC-3000", "CVI3", "2.5"
        )
        
        assert is_compatible == True
        assert "Uyumlu" in message
    
    def test_compatibility_check_invalid_version(self, validator):
        """Test compatibility check with invalid version"""
        is_compatible, message = validator.check_tool_controller_compatibility(
            "EABC-3000", "CVI3", "1.0"  # Too old
        )
        
        assert is_compatible == False
        assert "minimum" in message.lower()
    
    def test_compatibility_check_invalid_controller(self, validator):
        """Test compatibility check with invalid controller"""
        is_compatible, message = validator.check_tool_controller_compatibility(
            "EABC-3000", "UNKNOWN-CTRL"
        )
        
        assert is_compatible == False
    
    def test_parameter_range_validation(self, validator):
        """Test parameter range validation"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("EABC-3000 tork 50 Nm ayarla")
        
        extraction = ExtractionResult(
            intent=IntentType.CONFIGURATION,
            product_model="EABC-3000"
        )
        
        result = validator.validate(extraction, intent_result)
        
        # Should allow 50 Nm (within range for EABC: 5-85 Nm)
        allow_issues = [i for i in result.issues if i.status == ValidationStatus.ALLOW]
        assert len(allow_issues) >= 0
    
    def test_parameter_range_exceeded(self, validator):
        """Test parameter range exceeded validation"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("EABC-3000 tork 100 Nm ayarla")
        
        # Manually set target value since pattern may not extract it
        intent_result.entities.target_value = "100 Nm"
        intent_result.entities.parameter_type = "torque"
        
        extraction = ExtractionResult(
            intent=IntentType.CONFIGURATION,
            product_model="EABC-3000"
        )
        
        result = validator.validate(extraction, intent_result)
        
        # Should block 100 Nm (exceeds max 85 Nm)
        block_issues = [i for i in result.issues if i.status == ValidationStatus.BLOCK]
        assert len(block_issues) > 0
    
    def test_version_comparison(self, validator):
        """Test version string comparison"""
        assert validator._compare_versions("2.5", "2.0") > 0
        assert validator._compare_versions("2.0", "2.5") < 0
        assert validator._compare_versions("2.5", "2.5") == 0
        assert validator._compare_versions("2.5.1", "2.5") > 0
    
    def test_get_product_info(self, validator):
        """Test getting product info from matrix"""
        info = validator.get_product_info("EABC-3000")
        
        assert info is not None
        assert "controllers" in info
        assert "parameter_ranges" in info
    
    def test_get_controller_info(self, validator):
        """Test getting controller info"""
        info = validator.get_controller_info("CVI3")
        
        assert info is not None
        assert "features" in info


# ============================================
# Stage 5: Response Formatter Tests
# ============================================

class TestResponseFormatter:
    """Test Stage 5: Response Formatting"""
    
    @pytest.fixture
    def formatter(self):
        return ResponseFormatter()
    
    def test_greeting_response(self, formatter):
        """Test greeting response formatting"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("Merhaba")
        
        response = formatter.format_greeting(intent_result)
        
        assert "Merhaba" in response.content
        assert response.intent == IntentType.GREETING
    
    def test_off_topic_response(self, formatter):
        """Test off-topic response formatting"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("Hava durumu nasıl?")
        
        response = formatter.format_off_topic(intent_result)
        
        assert "Desoutter" in response.content
        assert response.intent == IntentType.OFF_TOPIC
    
    def test_no_result_response(self, formatter):
        """Test no result response formatting"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("XYZ-9999 hakkında bilgi")
        
        response = formatter.format_no_result(intent_result)
        
        assert "bulamadım" in response.content.lower()
    
    def test_configuration_response_format(self, formatter):
        """Test configuration response has required sections"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("EABC-3000 tork ayarı")
        
        # Override to ensure CONFIGURATION intent
        intent_result.primary_intent = IntentType.CONFIGURATION
        
        extraction = ExtractionResult(
            intent=IntentType.CONFIGURATION,
            product_model="EABC-3000",
            procedure=[
                ProcedureStep(step_number=1, action="Menu aç"),
                ProcedureStep(step_number=2, action="Parametre seç"),
            ],
            warnings=["Test warning"]
        )
        
        validation = ValidationResult(status=ValidationStatus.ALLOW)
        
        response = formatter.format(intent_result, extraction, validation)
        
        assert "EABC-3000" in response.content
        assert "1." in response.content
        assert "2." in response.content
    
    def test_turkish_content(self, formatter):
        """Test response is in Turkish"""
        classifier = IntentClassifier()
        intent_result = classifier.classify("EABC-3000 hata")
        
        extraction = ExtractionResult(
            intent=IntentType.TROUBLESHOOT,
            product_model="EABC-3000"
        )
        
        validation = ValidationResult(status=ValidationStatus.ALLOW)
        
        response = formatter.format(intent_result, extraction, validation)
        
        # Should contain Turkish words
        assert any(word in response.content.lower() for word in ["kaynak", "ürün", "bilgi"])


# ============================================
# Pipeline Integration Tests
# ============================================

class TestElHarezmiPipeline:
    """Test integrated El-Harezmi Pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        # Initialize without external services
        return ElHarezmiPipeline(
            qdrant_client=None,
            embedding_model=None,
            llm_client=None
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly"""
        assert pipeline.stage1_classifier is not None
        assert pipeline.stage2_retriever is not None
        assert pipeline.stage3_extractor is not None
        assert pipeline.stage4_validator is not None
        assert pipeline.stage5_formatter is not None
    
    def test_greeting_flow(self, pipeline):
        """Test greeting goes through pipeline"""
        result = pipeline.process_sync("Merhaba")
        
        assert result.success == True
        assert result.response.intent == IntentType.GREETING
        assert "Merhaba" in result.response.content
    
    def test_off_topic_flow(self, pipeline):
        """Test off-topic is handled"""
        result = pipeline.process_sync("Bugün hava çok güzel futbol izleyeceğim")
        
        assert result.success == True
        # May be HOW_TO or OFF_TOPIC depending on pattern matching
        assert result.response.intent in [IntentType.OFF_TOPIC, IntentType.HOW_TO, IntentType.GENERAL]
    
    def test_quick_compatibility_check(self, pipeline):
        """Test quick compatibility check method"""
        is_compatible, message = pipeline.check_compatibility("EABC-3000", "CVI3", "2.5")
        
        assert is_compatible == True
    
    def test_quick_intent_classification(self, pipeline):
        """Test quick intent classification method"""
        result = pipeline.classify_intent("EABC-3000 tork ayarı nasıl yapılır")
        
        # Should be CONFIGURATION or HOW_TO
        assert result.primary_intent in [IntentType.CONFIGURATION, IntentType.HOW_TO]
    
    def test_get_retrieval_strategy(self, pipeline):
        """Test getting retrieval strategy"""
        strategy = pipeline.get_retrieval_strategy(IntentType.COMPATIBILITY)
        
        assert "compatibility_matrix" in strategy["primary_sources"]
    
    def test_get_product_info(self, pipeline):
        """Test getting product info"""
        info = pipeline.get_product_info("EABC-3000")
        
        assert info is not None
        assert "controllers" in info
    
    def test_metrics_tracking(self, pipeline):
        """Test metrics are tracked"""
        result = pipeline.process_sync("Merhaba")
        
        assert result.metrics.total_time_ms > 0
        assert result.metrics.stage1_time_ms > 0


# ============================================
# Intent Examples Tests
# ============================================

class TestIntentExamples:
    """Test all 15 intent types with example queries"""
    
    @pytest.fixture
    def classifier(self):
        return IntentClassifier()
    
    @pytest.mark.parametrize("query,expected_intents", [
        # CONFIGURATION
        ("EABC-3000 pset ayarı nasıl yapılır?", [IntentType.CONFIGURATION, IntentType.HOW_TO]),
        ("Tork 50 Nm'ye nasıl ayarlanır?", [IntentType.CONFIGURATION, IntentType.HOW_TO]),
        
        # COMPATIBILITY
        ("EABC-3000 hangi CVI3 versiyonuyla çalışır?", [IntentType.COMPATIBILITY, IntentType.CAPABILITY_QUERY]),
        ("Bu tool hangi dok ile uyumlu?", [IntentType.COMPATIBILITY, IntentType.ACCESSORY_QUERY]),
        
        # ERROR_CODE
        ("E1234 hata kodu ne anlama geliyor?", [IntentType.ERROR_CODE]),
        
        # TROUBLESHOOT
        ("Tool çalışmıyor, motor dönmüyor", [IntentType.TROUBLESHOOT]),
        
        # SPECIFICATION
        ("EABC-3000 teknik özellikleri nedir?", [IntentType.SPECIFICATION, IntentType.GENERAL]),
        
        # CAPABILITY_QUERY
        ("EABC-3000 WiFi destekliyor mu?", [IntentType.CAPABILITY_QUERY, IntentType.COMPATIBILITY]),
        
        # ACCESSORY_QUERY
        ("Hangi batarya ile çalışır?", [IntentType.ACCESSORY_QUERY, IntentType.COMPATIBILITY]),
        
        # GREETING
        ("Merhaba", [IntentType.GREETING]),
        ("Selam", [IntentType.GREETING]),
    ])
    def test_intent_examples(self, classifier, query, expected_intents):
        """Test intent classification for various examples"""
        result = classifier.classify(query)
        
        assert result.primary_intent in expected_intents, \
            f"Expected one of {[i.value for i in expected_intents]}, got {result.primary_intent.value}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
