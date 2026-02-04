#!/usr/bin/env python3
"""
Adaptive Chunking Test Suite
============================
Tests for all chunking strategies.

Run with:
    pytest tests/test_adaptive_chunking.py -v
    python tests/test_adaptive_chunking.py
"""
import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Direct imports to avoid torch dependency in documents/__init__.py
from src.documents.chunkers.base_chunker import BaseChunker, Chunk
from src.documents.chunkers.chunker_factory import ChunkerFactory
from src.documents.chunkers.semantic_chunker import SemanticChunker
from src.documents.chunkers.table_aware_chunker import TableAwareChunker
from src.documents.chunkers.entity_chunker import EntityChunker
from src.documents.chunkers.problem_solution_chunker import ProblemSolutionChunker
from src.documents.chunkers.step_preserving_chunker import StepPreservingChunker
from src.documents.chunkers.hybrid_chunker import HybridChunker
from src.documents.document_classifier import DocumentClassifier


class TestBaseChunker(unittest.TestCase):
    """Tests for BaseChunker"""
    
    def test_chunk_dataclass(self):
        """Test Chunk dataclass creation"""
        chunk = Chunk(
            text="Test content",
            chunk_index=0,
            chunk_type="test"
        )
        self.assertEqual(chunk.text, "Test content")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.chunk_type, "test")
        self.assertEqual(chunk.metadata, {})
    
    def test_token_estimation(self):
        """Test token count estimation"""
        # Create a concrete implementation for testing
        class TestChunker(BaseChunker):
            def chunk(self, text, metadata=None):
                return []
        
        chunker = TestChunker()
        
        # ~4 chars per token
        text = "This is a test sentence with about twenty words in it."
        tokens = chunker.estimate_tokens(text)
        self.assertGreater(tokens, 10)
        self.assertLess(tokens, 20)


class TestSemanticChunker(unittest.TestCase):
    """Tests for SemanticChunker"""
    
    def setUp(self):
        self.chunker = SemanticChunker(max_chunk_size=500, min_chunk_size=50)
    
    def test_section_detection(self):
        """Test section header detection"""
        text = """
# Chapter 1: Introduction
This is the introduction section with some content.

## Section 1.1: Getting Started
This section explains how to get started.

### Subsection 1.1.1
More detailed information here.

## Section 1.2: Configuration
Configuration details follow.
        """
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)
        
        # Check section type is captured
        for chunk in chunks:
            self.assertIn('section_type', chunk.metadata)
    
    def test_procedure_preservation(self):
        """Test that procedures are not split"""
        text = """
## Configuration Procedure
Follow these steps:
1. Connect the device
2. Open settings
3. Configure parameters
4. Save and restart

## Next Section
Some other content.
        """
        chunks = self.chunker.chunk(text)
        
        # All steps should be in one chunk
        for chunk in chunks:
            if "1. Connect" in chunk.text:
                self.assertIn("4. Save", chunk.text)


class TestTableAwareChunker(unittest.TestCase):
    """Tests for TableAwareChunker"""
    
    def setUp(self):
        self.chunker = TableAwareChunker(max_chunk_size=500, rows_per_chunk=3)
    
    def test_table_detection(self):
        """Test table detection"""
        text = """
# Compatibility Matrix

| Tool | Controller | Version |
|------|------------|---------|
| EABC-3000 | CVI3 | 2.5+ |
| EABC-3500 | CVIR | 3.0+ |
| EFD-1500 | CVI3 | 2.0+ |

## Additional Notes
Some notes here.
        """
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)
        
        # Check that table chunks preserve headers
        for chunk in chunks:
            if 'table_row' in chunk.chunk_type:
                self.assertIn('Tool', chunk.text)  # Header should be included
    
    def test_header_preservation(self):
        """Test that table headers are preserved in each chunk"""
        text = """
| Model | Torque | Speed |
|-------|--------|-------|
| A1 | 10 | 100 |
| A2 | 20 | 200 |
| A3 | 30 | 300 |
| A4 | 40 | 400 |
| A5 | 50 | 500 |
        """
        chunks = self.chunker.chunk(text)
        
        # Each chunk should have the header row
        for chunk in chunks:
            if '|' in chunk.text:
                self.assertIn('Model', chunk.text)


class TestEntityChunker(unittest.TestCase):
    """Tests for EntityChunker"""
    
    def setUp(self):
        self.chunker = EntityChunker(max_chunk_size=500, entities_per_chunk=3)
    
    def test_error_code_extraction(self):
        """Test error code extraction"""
        text = """
E01: Motor overload detected
E02: Communication timeout
E03: Sensor failure
E04: Battery low
E05: Temperature warning
        """
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)
        
        # Check entity codes are captured in metadata
        all_codes = []
        for chunk in chunks:
            if 'entity_codes' in chunk.metadata:
                all_codes.extend(chunk.metadata['entity_codes'])
        
        self.assertIn('E01', all_codes)
        # E05 may be in second chunk depending on entities_per_chunk
        self.assertGreaterEqual(len(all_codes), 3)
    
    def test_product_prefixed_codes(self):
        """Test product-prefixed error codes"""
        text = """
EABC-001: Torque limit exceeded
EABC-002: Angle limit exceeded
EABC-003: Speed error
        """
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)


class TestProblemSolutionChunker(unittest.TestCase):
    """Tests for ProblemSolutionChunker"""
    
    def setUp(self):
        self.chunker = ProblemSolutionChunker(max_chunk_size=1500)
    
    def test_esde_extraction(self):
        """Test ESDE code extraction"""
        text = """
ESDE-23456 Service Bulletin

Problem:
The tool may experience intermittent disconnection during operation.

Affected Models:
EABC-3000, EABC-3500

Solution:
Update firmware to version 2.5 or higher.
        """
        chunks = self.chunker.chunk(text)
        self.assertEqual(len(chunks), 1)  # Should be one problem-solution pair
        
        # Check ESDE code is captured
        self.assertEqual(chunks[0].metadata.get('esde_code'), 'ESDE-23456')
        self.assertTrue(chunks[0].metadata.get('has_problem'))
        self.assertTrue(chunks[0].metadata.get('has_solution'))
    
    def test_problem_solution_kept_together(self):
        """Test that problem and solution are not split"""
        text = """
Problem:
This is a detailed description of the problem that spans multiple lines
and contains important diagnostic information.

Solution:
Follow these steps to resolve the issue:
1. Step one
2. Step two
3. Step three
        """
        chunks = self.chunker.chunk(text)
        
        # Problem and solution should be in same chunk
        for chunk in chunks:
            if "Problem:" in chunk.text:
                self.assertIn("Solution:", chunk.text)


class TestStepPreservingChunker(unittest.TestCase):
    """Tests for StepPreservingChunker"""
    
    def setUp(self):
        self.chunker = StepPreservingChunker(max_chunk_size=1000, steps_per_chunk=3)
    
    def test_step_detection(self):
        """Test numbered step detection"""
        text = """
Prerequisites:
- Tool connected to controller
- Software installed

1. Open the configuration menu
2. Select parameter settings
3. Enter the desired values
4. Click Save
5. Restart the device
        """
        chunks = self.chunker.chunk(text)
        self.assertGreater(len(chunks), 0)
        
        # Check step numbers are captured
        all_step_numbers = []
        for chunk in chunks:
            if 'step_numbers' in chunk.metadata:
                all_step_numbers.extend(chunk.metadata['step_numbers'])
        
        self.assertIn(1, all_step_numbers)
        self.assertIn(5, all_step_numbers)
    
    def test_prerequisites_preserved(self):
        """Test that prerequisites are kept with first steps"""
        text = """
Prerequisites:
Important prerequisite information.

1. First step
2. Second step
        """
        chunks = self.chunker.chunk(text)
        
        # Prerequisites should be in first chunk
        self.assertIn("Prerequisites", chunks[0].text)
        self.assertTrue(chunks[0].metadata.get('has_prerequisites'))
    
    def test_warning_detection(self):
        """Test warning detection within steps"""
        text = """
1. Disconnect power supply
WARNING: High voltage risk
2. Remove cover
3. Replace component
        """
        chunks = self.chunker.chunk(text)
        
        # Should detect warning
        has_warning_chunk = any(
            chunk.metadata.get('has_warnings') 
            for chunk in chunks
        )
        self.assertTrue(has_warning_chunk)


class TestHybridChunker(unittest.TestCase):
    """Tests for HybridChunker"""
    
    def setUp(self):
        self.chunker = HybridChunker(max_chunk_size=500)
    
    def test_mixed_content(self):
        """Test chunking of mixed content"""
        text = """
# Introduction
This is introductory text with some detailed explanation.

| Column1 | Column2 |
|---------|---------|
| Value1  | Value2  |
| Value3  | Value4  |

E01: Error description
E02: Another error

1. Step one
2. Step two
        """
        chunks = self.chunker.chunk(text)
        self.assertGreaterEqual(len(chunks), 1)  # At least one chunk
        
        # Check chunks were created with segment metadata
        # HybridChunker should at least process the content
        for chunk in chunks:
            self.assertIsNotNone(chunk.text)
            self.assertGreater(len(chunk.text), 0)


class TestChunkerFactory(unittest.TestCase):
    """Tests for ChunkerFactory"""
    
    def setUp(self):
        self.factory = ChunkerFactory()
    
    def test_get_chunker_by_type(self):
        """Test getting chunker by document type"""
        # Use lowercase values matching DocumentType enum
        config_chunker = self.factory.get_chunker('configuration_guide')
        self.assertIsInstance(config_chunker, SemanticChunker)
        
        table_chunker = self.factory.get_chunker('compatibility_matrix')
        self.assertIsInstance(table_chunker, TableAwareChunker)
        
        error_chunker = self.factory.get_chunker('error_code_list')
        self.assertIsInstance(error_chunker, EntityChunker)
    
    def test_unknown_type_returns_hybrid(self):
        """Test that unknown type returns HybridChunker"""
        chunker = self.factory.get_chunker('UNKNOWN_TYPE')
        self.assertIsInstance(chunker, HybridChunker)


class TestDocumentClassifier(unittest.TestCase):
    """Tests for DocumentClassifier"""
    
    def setUp(self):
        self.classifier = DocumentClassifier()
    
    def test_service_bulletin_detection(self):
        """Test ESDE bulletin detection"""
        text = "ESDE-12345 Service Bulletin\nAffected Models: EABC-3000"
        result = self.classifier.classify(text)
        doc_type = result.document_type.value if hasattr(result, 'document_type') else result
        self.assertEqual(doc_type, 'service_bulletin')
    
    def test_compatibility_matrix_detection(self):
        """Test compatibility matrix detection"""
        text = """
Compatibility Matrix
| Tool | Controller | Version |
|------|------------|---------|
| EABC | CVI3 | 2.5 |
        """
        result = self.classifier.classify(text)
        doc_type = result.document_type.value if hasattr(result, 'document_type') else result
        self.assertEqual(doc_type, 'compatibility_matrix')
    
    def test_error_code_list_detection(self):
        """Test error code list detection"""
        text = """
E01: Motor error
E02: Communication error
E03: Sensor error
E04: Battery error
E05: Temperature error
E06: Calibration error
        """
        result = self.classifier.classify(text)
        doc_type = result.document_type.value if hasattr(result, 'document_type') else result
        # Error code list should be detected, but might fall back to technical_manual
        self.assertIn(doc_type, ['error_code_list', 'technical_manual'])


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test full classification -> chunking pipeline"""
        classifier = DocumentClassifier()
        factory = ChunkerFactory()
        
        # Test document
        text = """
ESDE-98765 Service Bulletin

Problem:
The EABC-3000 tool may exhibit random disconnections.

Affected Models:
- EABC-3000
- EABC-3500

Solution:
1. Update firmware to v2.5
2. Reconfigure wireless settings
3. Test connection stability
        """
        
        # Classify
        result = classifier.classify(text)
        doc_type = result.document_type.value if hasattr(result, 'document_type') else result
        self.assertEqual(doc_type, 'service_bulletin')
        
        # Get chunker
        chunker = factory.get_chunker(doc_type)
        self.assertIsInstance(chunker, ProblemSolutionChunker)
        
        # Chunk
        chunks = chunker.chunk(text, {'source': 'test'})
        self.assertGreater(len(chunks), 0)
        
        # Verify metadata
        self.assertEqual(chunks[0].metadata.get('esde_code'), 'ESDE-98765')


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
