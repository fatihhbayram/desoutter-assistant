"""
Document Type Classifier
========================
Automatically detects document type for adaptive chunking strategy selection.

Document Types (8):
- TECHNICAL_MANUAL: Product manuals with chapters/sections
- SERVICE_BULLETIN: ESDE bulletins with known issues
- CONFIGURATION_GUIDE: Setup and parameter guides
- COMPATIBILITY_MATRIX: Tool-controller compatibility tables + CHANGELOGs
- SPEC_SHEET: Technical specifications
- ERROR_CODE_LIST: Error code definitions
- PROCEDURE_GUIDE: Step-by-step instructions
- FRESHDESK_TICKET: Support tickets

Usage:
    classifier = DocumentClassifier()
    doc_type, confidence = classifier.classify(content, filename)
    chunker = ChunkerFactory.get_chunker(doc_type)
"""
import re
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Document type classification for adaptive chunking"""
    TECHNICAL_MANUAL = "technical_manual"
    SERVICE_BULLETIN = "service_bulletin"
    CONFIGURATION_GUIDE = "configuration_guide"
    COMPATIBILITY_MATRIX = "compatibility_matrix"  # Includes CHANGELOGs
    SPEC_SHEET = "spec_sheet"
    ERROR_CODE_LIST = "error_code_list"
    PROCEDURE_GUIDE = "procedure_guide"
    FRESHDESK_TICKET = "freshdesk_ticket"


@dataclass
class ClassificationResult:
    """Result of document classification"""
    document_type: DocumentType
    confidence: float
    matched_patterns: List[str]
    detected_features: Dict[str, bool]


# =============================================================================
# CLASSIFICATION PATTERNS
# =============================================================================

DOCUMENT_TYPE_PATTERNS = {
    DocumentType.SERVICE_BULLETIN: {
        "required": [
            r"ESDE[-\s]?\d{4,5}",           # ESDE-12345 or ESDE 12345
            r"Service\s+Bulletin",           # Service Bulletin
            r"Servis\s+Bülteni",             # Turkish: Service Bulletin
            r"Technical\s+Bulletin",         # Technical Bulletin
            r"ESB[-\s]?\d+",                 # ESB-123
        ],
        "optional": [
            r"Affected\s+Models",            # Affected Models section
            r"Manufacturing\s+Defect",       # Manufacturing defect
            r"Root\s+Cause",                 # Root Cause Analysis
            r"Corrective\s+Action",          # Corrective Action
            r"Serial\s+Number\s+Range",      # Affected serial range
        ],
        "min_required": 1,
        "boost_if_filename": ["esde", "esb", "bulletin", "bulten"],
    },
    
    DocumentType.CONFIGURATION_GUIDE: {
        "required": [
            r"Configuration\s+Guide",
            r"Setup\s+Guide",
            r"Parameter\s+Settings?",
            r"Konfigürasyon",
            r"Ayar\s+Kılavuzu",
        ],
        "optional": [
            r"Pset\s+\d+",                   # Pset configuration
            r"Torque\s+Setting",
            r"Angle\s+Setting",
            r"Program\s+\d+",
            r"Target\s+Value",
        ],
        "min_required": 1,
        "boost_if_filename": ["config", "setup", "settings", "ayar"],
    },
    
    DocumentType.COMPATIBILITY_MATRIX: {
        "required": [
            r"Compatibility\s+Matrix",
            r"Uyumluluk\s+Matrisi",
            r"\|\s*Tool\s*\|.*Controller",   # Table header pattern
            r"Compatible\s+Controllers?",
            r"Supported\s+Versions?",
            # CHANGELOG patterns (merged)
            r"CHANGELOG",                     # Filename pattern
            r"Change\s*Log",                  # Change Log header
            r"Release\s+Notes?",              # Release Notes
            r"Version\s+History",             # Version History
            r"Sürüm\s+Geçmişi",               # Turkish: Version History
        ],
        "optional": [
            r"Firmware\s+Version",
            r"Min\s+Version",
            r"Required\s+Version",
            r"✓|✗|Yes|No|Evet|Hayır",        # Compatibility indicators
            # CHANGELOG optional patterns
            r"Previous\s+release",            # Version comparison
            r"Latest\s+release",              # Current version
            r"Products?\s+[Cc]ompatibility",  # Compatibility section
            r"Software\s+release",            # Software versions
            r"Controller\s+release",          # Controller versions
            r"[CAFV]\d+\.\d+\.\d+",           # Version patterns (C7.9.1, A7.8.2)
            r"Firmware\s+[Cc]ompatibility",   # Firmware compatibility
            r"Release\s+date",                # Release date column
        ],
        "min_required": 1,
        "boost_if_filename": ["compat", "matrix", "uyumluluk", "changelog", "release", "version"],
    },
    
    DocumentType.ERROR_CODE_LIST: {
        "required": [
            r"[A-Z]{1,4}[-\s]?\d{2,4}\s*:\s*", # Error code pattern (E01:, EABC-001:)
            r"Error\s+Code\s+List",
            r"Hata\s+Kodu\s+Listesi",
            r"Fault\s+Code",
            r"Diagnostic\s+Code",
        ],
        "optional": [
            r"Cause\s*:",
            r"Solution\s*:",
            r"Remedy\s*:",
            r"Description\s*:",
        ],
        "min_required": 2,  # Need at least 2 error code patterns
        "boost_if_filename": ["error", "fault", "hata", "diagnostic"],
    },
    
    DocumentType.PROCEDURE_GUIDE: {
        "required": [
            r"Procedure\s+Guide",
            r"Step\s+\d+\s*[:\.]",            # Step 1: or Step 1.
            r"^\s*\d+\.\s+[A-Z]",             # Numbered steps
            r"Installation\s+Procedure",
            r"Prosedür\s+Kılavuzu",
        ],
        "optional": [
            r"Prerequisites?:",
            r"Before\s+you\s+begin",
            r"Required\s+Tools",
            r"Warning\s*:|Caution\s*:",
            r"⚠️|🔧|⚡",                       # Emoji indicators
        ],
        "min_required": 2,
        "boost_if_filename": ["procedure", "install", "prosedur", "kurulum"],
    },
    
    DocumentType.SPEC_SHEET: {
        "required": [
            r"Specifications?",
            r"Technical\s+Data",
            r"Teknik\s+Özellikler",
            r"Product\s+Specifications?",
        ],
        "optional": [
            r"\d+\.?\d*\s*(Nm|kg|mm|rpm|bar|V|A|W)",  # Units
            r"Dimension",
            r"Weight",
            r"Power\s+Consumption",
            r"Rated\s+Torque",
        ],
        "min_required": 1,
        "boost_if_filename": ["spec", "data", "sheet", "ozellik"],
    },
    
    DocumentType.FRESHDESK_TICKET: {
        "required": [
            r"Ticket\s*#?\d+",
            r"Case\s*#?\d+",
            r"Agent\s+Response",
            r"Customer\s+Reply",
            r"Status:\s*(Open|Closed|Resolved|Pending)",
        ],
        "optional": [
            r"Priority:\s*(Low|Medium|High|Urgent)",
            r"Created\s+Date",
            r"Resolution\s+Notes?",
            r"Support\s+Agent",
        ],
        "min_required": 1,
        "boost_if_filename": ["ticket", "case", "support", "freshdesk"],
    },
    
    DocumentType.TECHNICAL_MANUAL: {
        "required": [
            r"User\s+Manual",
            r"Technical\s+Manual",
            r"Kullanım\s+Kılavuzu",
            r"Product\s+Manual",
            r"Chapter\s+\d+",
            r"Section\s+\d+\.\d+",
        ],
        "optional": [
            r"Table\s+of\s+Contents",
            r"Introduction",
            r"Safety\s+Information",
            r"Maintenance",
            r"Troubleshooting",
        ],
        "min_required": 1,
        "boost_if_filename": ["manual", "guide", "kilavuz", "handbook"],
    },
}


# =============================================================================
# CHUNKING STRATEGY MAPPING
# =============================================================================

CHUNKING_STRATEGIES = {
    DocumentType.TECHNICAL_MANUAL: {
        "strategy": "SemanticChunker",
        "max_chunk_size": 1000,
        "split_by": "section_headers",
        "preserve": ["procedures", "parameter_tables"],
        "intent_relevance": ["GENERAL", "TROUBLESHOOTING", "MAINTENANCE"],
    },
    
    DocumentType.SERVICE_BULLETIN: {
        "strategy": "ProblemSolutionChunker",
        "max_chunk_size": 1500,
        "chunk_unit": "problem_solution_pair",
        "preserve": ["esde_code", "affected_models", "problem", "solution"],
        "intent_relevance": ["TROUBLESHOOTING", "ERROR_CODE"],
    },
    
    DocumentType.CONFIGURATION_GUIDE: {
        "strategy": "SemanticChunker",
        "max_chunk_size": 1000,
        "split_by": "section_headers",
        "preserve": ["procedures", "parameter_tables"],
        "intent_relevance": ["CONFIGURATION", "PROCEDURE"],
    },
    
    DocumentType.COMPATIBILITY_MATRIX: {
        "strategy": "TableAwareChunker",
        "max_chunk_size": 1500,
        "chunk_unit": "table_row",
        "preserve": ["headers", "row_context", "version_tables", "compatibility_sections"],
        "extract_metadata": {
            "product_versions": r"[CAFV]\d+\.\d+\.\d+",
            "release_dates": r"\d{2}/\d{2}/\d{4}",
            "product_names": r"(ExB\s*Com|ExB\s*Advanced|ExB\s*Flex|EABC|EPBC|EABA|EPBA|EAB|EPB|CVI3|CONNECT|CVIL2|AXON)"
        },
        "intent_relevance": ["COMPATIBILITY", "FIRMWARE", "SPECIFICATION"],
    },
    
    DocumentType.SPEC_SHEET: {
        "strategy": "SemanticChunker",
        "max_chunk_size": 500,
        "split_by": "section_headers",
        "preserve": ["tables", "specifications"],
        "intent_relevance": ["SPECIFICATION", "CAPABILITY"],
    },
    
    DocumentType.ERROR_CODE_LIST: {
        "strategy": "EntityChunker",
        "chunk_unit": "error_code",
        "pair": ["code", "description", "solution"],
        "intent_relevance": ["ERROR_CODE", "TROUBLESHOOTING"],
    },
    
    DocumentType.PROCEDURE_GUIDE: {
        "strategy": "StepPreservingChunker",
        "max_chunk_size": 1500,
        "split_only_at": "major_section_breaks",
        "preserve": ["numbered_steps", "warnings", "prerequisites"],
        "intent_relevance": ["PROCEDURE", "INSTALLATION", "FIRMWARE"],
    },
    
    DocumentType.FRESHDESK_TICKET: {
        "strategy": "ProblemSolutionChunker",
        "max_chunk_size": 1000,
        "chunk_unit": "ticket",
        "preserve": ["problem", "solution", "agent_notes"],
        "intent_relevance": ["TROUBLESHOOTING", "CONFIGURATION"],
    },
}


class DocumentClassifier:
    """
    Classify documents by type for adaptive chunking.
    
    Uses pattern matching and heuristics to determine document type.
    """
    
    def __init__(self):
        """Initialize classifier with compiled patterns"""
        self.patterns = {}
        
        for doc_type, config in DOCUMENT_TYPE_PATTERNS.items():
            self.patterns[doc_type] = {
                "required": [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                            for p in config["required"]],
                "optional": [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                            for p in config.get("optional", [])],
                "min_required": config.get("min_required", 1),
                "boost_if_filename": config.get("boost_if_filename", []),
            }
        
        logger.info(f"DocumentClassifier initialized with {len(self.patterns)} types")
    
    def classify(
        self,
        content: str,
        filename: str = ""
    ) -> ClassificationResult:
        """
        Classify document based on content and filename.
        
        Args:
            content: Document text content
            filename: Original filename (optional but improves accuracy)
            
        Returns:
            ClassificationResult with type, confidence, and matched patterns
        """
        filename_lower = filename.lower()
        content_preview = content[:5000]  # Check first 5000 chars for speed
        
        scores = {}
        match_details = {}
        
        for doc_type, patterns in self.patterns.items():
            score = 0.0
            matched = []
            
            # Check required patterns
            required_matches = 0
            for pattern in patterns["required"]:
                matches = pattern.findall(content_preview)
                if matches:
                    required_matches += 1
                    matched.append(pattern.pattern)
                    score += 10  # Base score for required match
                    score += len(matches) * 2  # Bonus for multiple matches
            
            # Must meet minimum required threshold
            if required_matches < patterns["min_required"]:
                continue
            
            # Check optional patterns (bonus points)
            for pattern in patterns["optional"]:
                if pattern.search(content_preview):
                    score += 3
                    matched.append(pattern.pattern)
            
            # Filename boost
            for keyword in patterns["boost_if_filename"]:
                if keyword in filename_lower:
                    score *= 1.5
                    matched.append(f"filename:{keyword}")
            
            if score > 0:
                scores[doc_type] = score
                match_details[doc_type] = matched
        
        # Select best match
        if scores:
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            # Normalize confidence (0-1)
            max_possible = 100  # Rough max expected score
            confidence = min(best_score / max_possible, 1.0)
            
            # Detect features for metadata
            features = self._detect_features(content_preview)
            
            logger.info(
                f"[CLASSIFY] {filename} → {best_type.value} "
                f"(confidence: {confidence:.2f}, patterns: {len(match_details[best_type])})"
            )
            
            return ClassificationResult(
                document_type=best_type,
                confidence=confidence,
                matched_patterns=match_details[best_type],
                detected_features=features
            )
        
        # Fallback to TECHNICAL_MANUAL
        logger.info(f"[CLASSIFY] {filename} → technical_manual (default fallback)")
        return ClassificationResult(
            document_type=DocumentType.TECHNICAL_MANUAL,
            confidence=0.3,
            matched_patterns=[],
            detected_features=self._detect_features(content_preview)
        )
    
    def _detect_features(self, content: str) -> Dict[str, bool]:
        """Detect content features for metadata enrichment"""
        return {
            "contains_procedure": bool(re.search(
                r"(Step\s+\d+|^\s*\d+\.\s+[A-Z])", 
                content, re.IGNORECASE | re.MULTILINE
            )),
            "contains_table": bool(re.search(
                r"\|.*\|.*\||\t.*\t", 
                content
            )),
            "contains_error_code": bool(re.search(
                r"[A-Z]{1,4}[-\s]?\d{2,4}\s*:", 
                content
            )),
            "contains_warning": bool(re.search(
                r"(Warning|Caution|Danger|⚠️|Uyarı|Dikkat)", 
                content, re.IGNORECASE
            )),
            "contains_esde": bool(re.search(
                r"ESDE[-\s]?\d{4,5}", 
                content
            )),
            "contains_specs": bool(re.search(
                r"\d+\.?\d*\s*(Nm|kg|mm|rpm|bar|V|A|W)", 
                content
            )),
        }
    
    def get_chunking_strategy(self, doc_type: DocumentType) -> Dict:
        """
        Get recommended chunking strategy for document type.
        
        Args:
            doc_type: Classified document type
            
        Returns:
            Strategy configuration dict
        """
        return CHUNKING_STRATEGIES.get(doc_type, CHUNKING_STRATEGIES[DocumentType.TECHNICAL_MANUAL])
    
    def get_intent_relevance(self, doc_type: DocumentType) -> List[str]:
        """
        Get intent types this document type is relevant for.
        
        Args:
            doc_type: Classified document type
            
        Returns:
            List of intent names
        """
        strategy = self.get_chunking_strategy(doc_type)
        return strategy.get("intent_relevance", ["GENERAL"])


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_document_classifier: Optional[DocumentClassifier] = None


def get_document_classifier() -> DocumentClassifier:
    """Get singleton DocumentClassifier instance"""
    global _document_classifier
    
    if _document_classifier is None:
        _document_classifier = DocumentClassifier()
    
    return _document_classifier


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classifier = DocumentClassifier()
    
    # Test samples
    test_samples = [
        ("ESDE-23028 Service Bulletin\n\nAffected Models: EABC-3000\nRoot Cause: Manufacturing defect...", 
         "ESDE-23028.pdf", 
         DocumentType.SERVICE_BULLETIN),
        
        ("User Manual\n\nChapter 1: Introduction\nSection 1.1: Safety Information...", 
         "EABC-3000_Manual.pdf", 
         DocumentType.TECHNICAL_MANUAL),
        
        ("Configuration Guide\n\nPset Settings:\nStep 1. Navigate to Menu\nStep 2. Set torque to 50 Nm...", 
         "CVI3_Config_Guide.pdf", 
         DocumentType.CONFIGURATION_GUIDE),
        
        ("Compatibility Matrix\n\n| Tool | Controller | Min Version |\n| EABC-3000 | CVI3 | 2.5 |...", 
         "Compatibility_Matrix.xlsx", 
         DocumentType.COMPATIBILITY_MATRIX),
        
        ("Error Code List\n\nE001: Motor overload\nE002: Communication failure\nE003: Torque sensor error...", 
         "ErrorCodes.pdf", 
         DocumentType.ERROR_CODE_LIST),
    ]
    
    print("=" * 70)
    print("DOCUMENT CLASSIFIER TEST")
    print("=" * 70)
    
    passed = 0
    for content, filename, expected in test_samples:
        result = classifier.classify(content, filename)
        is_correct = result.document_type == expected
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n{status} {filename}")
        print(f"   Expected: {expected.value}")
        print(f"   Got: {result.document_type.value} (conf: {result.confidence:.2f})")
        print(f"   Patterns: {result.matched_patterns[:3]}")
        print(f"   Features: {result.detected_features}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(test_samples)} passed ({passed/len(test_samples)*100:.0f}%)")
    print("=" * 70)
