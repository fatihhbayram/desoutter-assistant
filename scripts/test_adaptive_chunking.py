
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.documents.semantic_chunker import SemanticChunker, DocumentType

class TestAdaptiveChunking(unittest.TestCase):
    
    def setUp(self):
        # Default 400
        self.chunker = SemanticChunker(chunk_size=400, chunk_overlap=100)
        
        # Create a single MASSIVE paragraph to force splitting based on size
        # 1000+ words in one paragraph
        sentence = "This is a long sentence used for testing chunk size adaptation logic in the system to verify that the chunker correctly splits large blocks of text according to the defined limits. " * 5
        self.long_text = sentence * 20 # ~1500 words in one paragraph

    def test_manual_chunk_size(self):
        """Test standard size for MANUAL"""
        print("\nTesting Manual Chunking (Standard Size)...")
        chunks = self.chunker.chunk_document(
            text=self.long_text,
            source_filename="manual.pdf",
            doc_type=DocumentType.TECHNICAL_MANUAL
        )
        
        avg_size = sum(c['metadata']['word_count'] for c in chunks) / len(chunks)
        print(f"Average Chunk Size (Manual): {avg_size:.2f} words")
        
        # Manuals should use default 400 (approx 300-400 words)
        self.assertGreater(avg_size, 250, "Manual chunks should be relatively large")

    def test_troubleshooting_chunk_size(self):
        """Test smaller size for TROUBLESHOOTING"""
        print("\nTesting Troubleshooting Chunking (Small Size)...")
        chunks = self.chunker.chunk_document(
            text=self.long_text,
            source_filename="troubleshooting.pdf",
            doc_type=DocumentType.TROUBLESHOOTING_GUIDE
        )
        
        avg_size = sum(c['metadata']['word_count'] for c in chunks) / len(chunks)
        print(f"Average Chunk Size (Troubleshooting): {avg_size:.2f} words")
        
        # Troubleshooting should be ~200 (approx 150-200 words)
        # Should be significantly smaller than manual chunks
        self.assertLess(avg_size, 250, "Troubleshooting chunks should be smaller")
        
    def test_safety_chunk_size(self):
        """Test medium size for SAFETY"""
        print("\nTesting Safety Chunking (Medium Size)...")
        chunks = self.chunker.chunk_document(
            text=self.long_text,
            source_filename="safety.pdf",
            doc_type=DocumentType.SAFETY_DOCUMENT
        )
        avg_size = sum(c['metadata']['word_count'] for c in chunks) / len(chunks)
        print(f"Average Chunk Size (Safety): {avg_size:.2f} words")
        
        self.assertLess(avg_size, 300)

if __name__ == '__main__':
    unittest.main()
