"""
Query Intent Detection Module
Classifies user queries into specific intent categories to enable
context-appropriate response generation.

Intent Types:
- TROUBLESHOOTING: Error, fault, problem diagnosis
- SPECIFICATIONS: Torque, speed, dimensions, technical specs
- INSTALLATION: Setup, assembly, configuration
- CALIBRATION: Calibrate, adjust, zero setting
- MAINTENANCE: Clean, lubricate, replace parts
- CONNECTION: WiFi, ethernet, network, pairing
- ERROR_CODE: E01, E02, error code lookup
- GENERAL: Default for unclear intent
"""
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import re
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryIntent(str, Enum):
    """Query intent classification"""
    TROUBLESHOOTING = "troubleshooting"  # Error, fault, problem
    SPECIFICATIONS = "specifications"     # Torque, speed, weight, dimensions
    INSTALLATION = "installation"         # Setup, assembly, configuration
    CALIBRATION = "calibration"           # Calibrate, adjust, zero
    MAINTENANCE = "maintenance"           # Clean, lubricate, replace
    CONNECTION = "connection"             # WiFi, ethernet, network, pairing
    ERROR_CODE = "error_code"            # E01, E02, error code lookup
    GENERAL = "general"                   # Default for unclear intent


@dataclass
class IntentResult:
    """Result of intent detection"""
    intent: QueryIntent  # Primary detected intent
    confidence: float  # 0.0-1.0 confidence score
    matched_patterns: List[str]  # Patterns that matched
    secondary_intent: Optional[QueryIntent] = None  # Secondary intent if applicable


class IntentDetector:
    """
    Detect user query intent from patterns and keywords
    
    Uses multi-level detection:
    1. Explicit pattern matching (error codes, spec keywords)
    2. Contextual analysis (product type context)
    3. Query structure (question vs statement)
    """
    
    # Intent-specific keyword patterns
    INTENT_PATTERNS = {
        QueryIntent.ERROR_CODE: {
            'patterns': [
                r'\bE\d{2,3}\b',  # E01, E018, etc.
                r'\berror\s+code\b',
                r'\bfault\s+code\b',
                r'\balarm\s+\d+\b',
                r'\btransducer\s+error\b'
            ],
            'weight': 1.0  # Highest priority
        },
        
        QueryIntent.TROUBLESHOOTING: {
            'patterns': [
                r'\bnot\s+working\b',
                r'\b(problem|issue|fault|error)\b',
                r'\b(broken|damaged|failed)\b',
                r"\bdoesn['\"]?t\s+(work|start|run)\b",
                r'\b(stopped|stuck|jammed)\b',
                r'\b(noise|vibration|smell)\b',
                r'\b(overheating|overheat)\b',
                r'\bçalışmıyor\b',  # Turkish: not working
                r'\bsorun\b',  # Turkish: problem
                r'\barıza\b'  # Turkish: fault
            ],
            'weight': 0.9
        },
        
        QueryIntent.SPECIFICATIONS: {
            'patterns': [
                r'\b(torque|tork)\b',
                r'\b(speed|hız|rpm)\b',
                r'\b(weight|ağırlık|kg|grams?)\b',
                r'\b(dimensions?|boyutlar|size)\b',
                r'\b(capacity|kapasite)\b',
                r'\b(specification|özellik|spec)\b',
                r'\b(how\s+much|ne\s+kadar)\b',
                r'\bwhat\s+(is|are)\s+the\b',
                r'\b(power|güç|watt|voltage|volt)\b',
                r'\b(range|aralık|min|max)\b'
            ],
            'weight': 0.95
        },
        
        QueryIntent.INSTALLATION: {
            'patterns': [
                r'\b(install|kurulum|setup|set\s+up)\b',
                r'\b(assembly|montaj|assemble)\b',
                r'\b(configuration|configure|ayarla)\b',
                r'\b(connect|bağla|connection\s+setup)\b',
                r'\bhow\s+to\s+(install|setup|configure)\b',
                r'\b(mounting|mount|tak)\b',
                r'\b(initialization|initialize)\b'
            ],
            'weight': 0.85
        },
        
        QueryIntent.CALIBRATION: {
            'patterns': [
                r'\b(calibrat(e|ion)|kalibr(e|asyon))\b',
                r'\b(adjust|ayarla|tune)\b',
                r'\b(zero|sıfırla|reset)\b',
                r'\b(accuracy|doğruluk)\b',
                r'\b(torque\s+setting)\b',
                r'\bhow\s+to\s+(calibrate|adjust)\b'
            ],
            'weight': 0.9
        },
        
        QueryIntent.MAINTENANCE: {
            'patterns': [
                r'\b(maintenance|bakım|service)\b',
                r'\b(clean|temizle|cleaning)\b',
                r'\b(lubricate|yağla|oil)\b',
                r'\b(replace|değiştir|replacement)\b',
                r'\b(inspect|kontrol|inspection)\b',
                r'\b(preventive|önleyici)\b',
                r'\bhow\s+often\b'
            ],
            'weight': 0.8
        },
        
        QueryIntent.CONNECTION: {
            'patterns': [
                r'\b(wifi|wi-fi|wireless|kablosuz)\b',
                r'\b(network|ağ|ethernet)\b',
                r'\b(connection|bağlantı|connectivity)\b',
                r'\b(access\s+point|AP)\b',
                r'\b(pairing|eşleşme)\b',
                r'\b(connect\s+unit)\b',
                r'\b(IP\s+address|network\s+settings)\b',
                r'\b(cannot\s+connect|bağlanamıyor)\b'
            ],
            'weight': 0.9
        }
    }

    
    def __init__(self):
        """Initialize intent detector"""
        logger.info("IntentDetector initialized")
    
    def detect_intent(
        self,
        query: str,
        product_info: Optional[Dict] = None
    ) -> IntentResult:
        """
        Detect query intent using pattern matching
        
        Args:
            query: User query text
            product_info: Optional product information for context
        
        Returns:
            IntentResult with detected intent and confidence
        """
        query_lower = query.lower()
        
        # Score each intent
        intent_scores = {}
        matched_patterns_by_intent = {}
        
        for intent, config in self.INTENT_PATTERNS.items():
            score = 0.0
            matched = []
            
            for pattern in config['patterns']:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += config['weight']
                    matched.append(pattern)
            
            # Normalize score (0-1 range)
            if matched:
                intent_scores[intent] = min(score, 1.0)
                matched_patterns_by_intent[intent] = matched
        
        # If no patterns matched, default to GENERAL
        if not intent_scores:
            logger.info(f"No specific intent detected, defaulting to GENERAL")
            return IntentResult(
                intent=QueryIntent.GENERAL,
                confidence=0.5,
                matched_patterns=[],
                secondary_intent=None
            )
        
        # Get primary intent (highest score)
        primary_intent = max(intent_scores, key=intent_scores.get)
        primary_score = intent_scores[primary_intent]
        
        # Get secondary intent if exists
        remaining_intents = {k: v for k, v in intent_scores.items() if k != primary_intent}
        secondary_intent = None
        if remaining_intents:
            secondary_intent = max(remaining_intents, key=remaining_intents.get)
            # Only keep if score is significant
            if remaining_intents[secondary_intent] < 0.5:
                secondary_intent = None
        
        logger.info(
            f"Intent detected: {primary_intent.value} "
            f"(confidence={primary_score:.2f}, "
            f"patterns={len(matched_patterns_by_intent[primary_intent])})"
        )
        
        return IntentResult(
            intent=primary_intent,
            confidence=primary_score,
            matched_patterns=matched_patterns_by_intent[primary_intent],
            secondary_intent=secondary_intent
        )
    
    def get_intent_description(self, intent: QueryIntent) -> str:
        """
        Get human-readable description of intent
        
        Args:
            intent: Query intent
        
        Returns:
            Description string
        """
        descriptions = {
            QueryIntent.TROUBLESHOOTING: "Diagnose and fix problems",
            QueryIntent.SPECIFICATIONS: "Technical specifications and parameters",
            QueryIntent.INSTALLATION: "Setup and installation procedures",
            QueryIntent.CALIBRATION: "Calibration and adjustment",
            QueryIntent.MAINTENANCE: "Maintenance and service procedures",
            QueryIntent.CONNECTION: "Network and connectivity setup",
            QueryIntent.ERROR_CODE: "Error code lookup and resolution",
            QueryIntent.GENERAL: "General inquiry"
        }
        return descriptions.get(intent, "Unknown intent")