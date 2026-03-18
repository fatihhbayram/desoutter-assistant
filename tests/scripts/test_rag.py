#!/usr/bin/env python3
"""
Test RAG system with sample queries
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.rag_engine import RAGEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_rag():
    """Test RAG with sample queries"""
    logger.info("=" * 80)
    logger.info("🧪 Testing RAG System")
    logger.info("=" * 80)
    
    # Initialize RAG engine
    logger.info("\n🔧 Initializing RAG Engine...")
    rag = RAGEngine()
    
    # Test queries
    test_cases = [
        {
            "part_number": "6151659030",
            "fault": "Tool won't turn on, battery is charged",
            "language": "en"
        },
        {
            "part_number": "6151659030",
            "fault": "Aletin tork ayarı değişmiyor",
            "language": "tr"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n{'='  * 80}")
        logger.info(f"Test {i}/{len(test_cases)}")
        logger.info(f"{'=' * 80}")
        logger.info(f"Part Number: {test['part_number']}")
        logger.info(f"Fault: {test['fault']}")
        logger.info(f"Language: {test['language']}")
        logger.info("")
        
        # Generate suggestion
        result = rag.generate_repair_suggestion(
            part_number=test["part_number"],
            fault_description=test["fault"],
            language=test["language"]
        )
        
        # Display result
        logger.info(f"Confidence: {result['confidence']}")
        logger.info(f"Product: {result['product_model']}")
        logger.info(f"\nSources ({len(result['sources'])}):")
        for src in result['sources']:
            logger.info(f"  - {src['source']} (similarity: {src['similarity']})")
        
        logger.info(f"\n📝 Repair Suggestion:\n")
        print(result['suggestion'])
        print("")
    
    logger.info("=" * 80)
    logger.info("✅ RAG Test Completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_rag()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)
