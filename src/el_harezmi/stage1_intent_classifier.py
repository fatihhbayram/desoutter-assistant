"""
Stage 1: Intent Classification

Multi-label intent detection with entity extraction for Desoutter queries.

Responsibilities:
- Classify user queries into 1 primary + 0-2 secondary intents
- Extract entities: product_model, parameter_type, target_value, error_code, etc.
- Provide confidence scores for classification
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """15 Intent Types for Desoutter Technical Support"""
    
    # Core Intents (Original 8)
    TROUBLESHOOT = "troubleshoot"
    ERROR_CODE = "error_code"
    HOW_TO = "how_to"
    MAINTENANCE = "maintenance"
    GENERAL = "general"
    GREETING = "greeting"
    OFF_TOPIC = "off_topic"
    CLARIFICATION = "clarification"
    
    # New Intents (Expanded to 15)
    CONFIGURATION = "configuration"
    COMPATIBILITY = "compatibility"
    SPECIFICATION = "specification"
    PROCEDURE = "procedure"
    CALIBRATION = "calibration"
    FIRMWARE = "firmware"
    INSTALLATION = "installation"
    COMPARISON = "comparison"
    CAPABILITY_QUERY = "capability_query"
    ACCESSORY_QUERY = "accessory_query"


@dataclass
class ExtractedEntities:
    """Entities extracted from user query"""
    product_model: Optional[str] = None
    product_family: Optional[str] = None
    controller_type: Optional[str] = None
    error_code: Optional[str] = None
    parameter_type: Optional[str] = None
    target_value: Optional[str] = None
    firmware_version: Optional[str] = None
    accessory_type: Optional[str] = None
    query_objects: List[str] = field(default_factory=list)


@dataclass
class IntentResult:
    """Result of intent classification"""
    primary_intent: IntentType
    secondary_intents: List[IntentType] = field(default_factory=list)
    confidence: float = 0.0
    entities: ExtractedEntities = field(default_factory=ExtractedEntities)
    raw_query: str = ""
    normalized_query: str = ""


class IntentClassifier:
    """
    Stage 1: Multi-label Intent Classifier with Entity Extraction
    
    Uses pattern matching + keyword detection for fast, deterministic classification.
    Falls back to LLM for ambiguous queries.
    """
    
    # Intent keyword patterns (Turkish + English)
    INTENT_PATTERNS = {
        IntentType.ERROR_CODE: [
            # Error code keywords (not patterns - patterns are in PRODUCT_PATTERNS)
            r"hata\s*kodu", r"error\s*code", r"fault\s*code",
            r"hata\s*mesajı", r"error\s*message",
            r"arıza\s*kodu", r"diagnostic\s*code",
            r"hata\s*\d+", r"error\s*\d+",
            r"E\d{3,4}",  # Error codes like E001, E1234
        ],
        
        IntentType.TROUBLESHOOT: [
            r"çalışmıyor", r"arıza", r"sorun", r"problem",
            r"not\s*working", r"broken", r"failed", r"failure",
            r"doesn't\s*work", r"won't\s*start", r"stopped",
            r"neden\s*(?:olmuyor|çalışmıyor|açılmıyor)",
            r"what's\s*wrong", r"ne\s*oldu", r"bozuldu",
        ],
        
        IntentType.CONFIGURATION: [
            r"ayar(?:la|ı|ları)?", r"konfigür", r"config",
            r"pset", r"parametre", r"parameter",
            r"tork\s*ayar", r"torque\s*set",
            r"açı\s*ayar", r"angle\s*set",
            r"nasıl\s*ayar", r"how\s*to\s*set",
            r"değer(?:ini|i)?\s*(?:gir|ayarla|değiştir)",
        ],
        
        IntentType.COMPATIBILITY: [
            r"uyumlu\s*mu", r"compatible", r"compatibility",
            r"çalışır\s*mı", r"works\s*with",
            r"hangi\s*(?:controller|kontrol|cvi|dock)",
            r"which\s*(?:controller|version)",
            r"destekliyor\s*mu", r"support",
            r"ile\s*(?:çalışır|uyumlu)",
            r"version\s*(?:gerek|require)",
        ],
        
        IntentType.SPECIFICATION: [
            r"teknik\s*özellik", r"specification", r"spec",
            r"boyut", r"dimension", r"ağırlık", r"weight",
            r"güç\s*tüketim", r"power\s*consumption",
            r"kaç\s*(?:kg|mm|cm|volt|watt|amp)",
            r"ne\s*kadar\s*(?:ağır|büyük|uzun)",
        ],
        
        IntentType.PROCEDURE: [
            r"prosedür", r"procedure", r"adım(?:lar)?",
            r"step\s*by\s*step", r"sıra(?:sı|yla)",
            r"nasıl\s*yapılır", r"how\s*to\s*(?:do|perform)",
            r"talimat", r"instruction",
        ],
        
        IntentType.CALIBRATION: [
            r"kalibrasyon", r"calibration", r"calibrate",
            r"doğrula", r"verify", r"validation",
            r"hassasiyet", r"accuracy", r"precision",
        ],
        
        IntentType.FIRMWARE: [
            r"firmware", r"yazılım\s*güncelle",
            r"update", r"upgrade", r"downgrade",
            r"versiyon\s*(?:güncelle|yükselt)",
            r"software\s*update",
        ],
        
        IntentType.INSTALLATION: [
            r"kurulum", r"installation", r"install",
            r"montaj", r"mount", r"setup",
            r"ilk\s*(?:kurulum|ayar)", r"initial\s*setup",
            r"nasıl\s*(?:kurulur|monte\s*edilir)",
        ],
        
        IntentType.MAINTENANCE: [
            r"bakım", r"maintenance", r"temizlik",
            r"cleaning", r"yağlama", r"lubrication",
            r"servis\s*(?:aralığı|periyod)",
            r"preventive", r"önleyici",
        ],
        
        IntentType.HOW_TO: [
            r"nasıl", r"how\s*to", r"how\s*do\s*i",
            r"ne\s*şekilde", r"hangi\s*yöntem",
            r"yapabilir\s*miyim", r"can\s*i",
        ],
        
        IntentType.COMPARISON: [
            r"karşılaştır", r"compare", r"comparison",
            r"fark(?:ı|lar)?", r"difference",
            r"hangisi\s*(?:daha|iyi|tercih)",
            r"which\s*(?:is\s*better|one|should)",
            r"vs\.?|versus",
        ],
        
        IntentType.CAPABILITY_QUERY: [
            r"var\s*mı", r"does\s*it\s*have",
            r"destekliyor\s*mu", r"support",
            r"özelliği\s*var", r"feature",
            r"yapabilir\s*mi", r"can\s*it",
            r"mümkün\s*mü", r"possible",
            r"maksimum", r"maximum", r"minimum",
        ],
        
        IntentType.ACCESSORY_QUERY: [
            r"aksesuar", r"accessory", r"accessories",
            r"batarya", r"battery", r"pil",
            r"dok", r"dock", r"şarj",
            r"kablo", r"cable", r"adaptör", r"adapter",
            r"yedek\s*parça", r"spare\s*part",
        ],
        
        IntentType.GREETING: [
            r"^(?:merhaba|selam|hey|hi|hello|günaydın|iyi\s*günler)",
            r"nasılsın", r"how\s*are\s*you",
        ],
        
        IntentType.OFF_TOPIC: [
            r"hava\s*durumu", r"weather",
            r"yemek\s*tarifi", r"recipe",
            r"futbol|basketbol|spor",
            r"film|dizi|movie",
        ],
    }
    
    # Product patterns
    PRODUCT_PATTERNS = {
        "product_model": [
            # Match EABC-3000, EFD-1500, etc. - stop at word boundary (no extra suffixes)
            r"\b(E(?:ABC|FD|PBS|RSF|ABS|PBAHT)[-\s]?\d{3,5})\b",
            r"\b(CVI[RC3][-\s]?\d+)\b",
            r"\b(CVIC[-\s]?II)\b",
            r"\b(CONNECT)\b",
        ],
        "controller_type": [
            r"\b(CVI3|CVIR|CVIC[-\s]?II|CONNECT)\b",
        ],
        "error_code": [
            # Error codes that are NOT product models
            # Format: E + 3-4 digits (E001, E1234) or specific error prefixes
            r"\b(E\d{3,4})\b",  # E001, E1234
            r"\b(ERR[-\s]?\d{3,4})\b",  # ERR-001, ERR1234
            r"\b(FAULT[-\s]?\d{3,4})\b",  # FAULT-001
        ],
        "firmware_version": [
            r"\bv?(\d+\.\d+(?:\.\d+)?)\b",
        ],
    }
    
    # Parameter patterns
    PARAMETER_PATTERNS = {
        "torque": [r"tork|torque", r"(\d+(?:\.\d+)?)\s*(?:Nm|nm|N\.m)"],
        "angle": [r"açı|angle", r"(\d+(?:\.\d+)?)\s*(?:°|derece|degree)"],
        "speed": [r"hız|speed|rpm", r"(\d+)\s*(?:RPM|rpm|d/dk)"],
        "current": [r"akım|current", r"(\d+(?:\.\d+)?)\s*(?:A|amp)"],
    }
    
    # Product family mapping
    PRODUCT_FAMILIES = {
        "EFD": ["EFD", "Dispensing"],
        "EABC": ["EABC", "Advanced Battery"],
        "EPBAHT": ["EPBAHT", "Pneumatic"],
        "ERSF": ["ERSF", "RF"],
        "EABS": ["EABS", "Standard Battery"],
    }
    
    def __init__(self, confidence_threshold: float = 0.75):
        """
        Initialize Intent Classifier.
        
        Args:
            confidence_threshold: Minimum confidence to accept classification
        """
        self.confidence_threshold = confidence_threshold
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        self._compiled_intents = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._compiled_intents[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._compiled_products = {}
        for key, patterns in self.PRODUCT_PATTERNS.items():
            self._compiled_products[key] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self._compiled_params = {}
        for param, patterns in self.PARAMETER_PATTERNS.items():
            self._compiled_params[param] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify user query into intents and extract entities.
        
        Args:
            query: User query string
            
        Returns:
            IntentResult with primary/secondary intents and entities
        """
        normalized = self._normalize_query(query)
        
        # Score all intents
        intent_scores = self._score_intents(normalized)
        
        # Extract entities
        entities = self._extract_entities(query, normalized)
        
        # Determine primary and secondary intents
        primary, secondaries, confidence = self._determine_intents(intent_scores, entities)
        
        # Adjust intent based on entities
        primary, secondaries = self._adjust_by_entities(primary, secondaries, entities)
        
        result = IntentResult(
            primary_intent=primary,
            secondary_intents=secondaries,
            confidence=confidence,
            entities=entities,
            raw_query=query,
            normalized_query=normalized,
        )
        
        logger.info(f"Intent classified: {primary.value} (confidence={confidence:.2f})")
        if secondaries:
            logger.info(f"Secondary intents: {[s.value for s in secondaries]}")
        
        return result
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching"""
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Normalize Turkish characters for matching
        # (keep original for entity extraction)
        
        return normalized
    
    def _score_intents(self, query: str) -> Dict[IntentType, float]:
        """Score each intent based on pattern matches"""
        scores = {intent: 0.0 for intent in IntentType}
        
        for intent, patterns in self._compiled_intents.items():
            match_count = 0
            for pattern in patterns:
                matches = pattern.findall(query)
                if matches:
                    match_count += len(matches)
            
            if match_count > 0:
                # Score based on number of matches (diminishing returns)
                scores[intent] = min(1.0, 0.5 + (match_count * 0.2))
        
        return scores
    
    def _extract_entities(self, raw_query: str, normalized: str) -> ExtractedEntities:
        """Extract entities from query"""
        entities = ExtractedEntities()
        
        # Extract product model
        for pattern in self._compiled_products["product_model"]:
            match = pattern.search(raw_query)
            if match:
                entities.product_model = match.group(1).upper().replace(" ", "-")
                break
        
        # Extract controller type
        for pattern in self._compiled_products["controller_type"]:
            match = pattern.search(raw_query)
            if match:
                entities.controller_type = match.group(1).upper()
                break
        
        # Extract error code
        for pattern in self._compiled_products["error_code"]:
            match = pattern.search(raw_query)
            if match:
                entities.error_code = match.group(1).upper().replace(" ", "-")
                break
        
        # Extract firmware version
        for pattern in self._compiled_products["firmware_version"]:
            match = pattern.search(raw_query)
            if match:
                entities.firmware_version = match.group(1)
                break
        
        # Determine product family
        if entities.product_model:
            for family, prefixes in self.PRODUCT_FAMILIES.items():
                for prefix in prefixes:
                    if entities.product_model.startswith(prefix):
                        entities.product_family = family
                        break
        
        # Extract parameters and values
        for param_type, patterns in self._compiled_params.items():
            # Check if parameter type is mentioned
            if patterns[0].search(normalized):
                entities.parameter_type = param_type
                # Try to extract value
                if len(patterns) > 1:
                    match = patterns[1].search(raw_query)
                    if match:
                        entities.target_value = match.group(0)
                break
        
        # Extract accessory type
        accessory_keywords = {
            "batarya": "battery", "battery": "battery", "pil": "battery",
            "dok": "dock", "dock": "dock", "şarj": "dock",
            "kablo": "cable", "cable": "cable",
            "adaptör": "adapter", "adapter": "adapter",
        }
        for keyword, accessory in accessory_keywords.items():
            if keyword in normalized:
                entities.accessory_type = accessory
                break
        
        # Extract query objects (what user is asking about)
        query_objects = []
        object_keywords = ["controller", "kontrol", "tool", "alet", "dok", "dock", 
                          "batarya", "battery", "firmware", "yazılım"]
        for keyword in object_keywords:
            if keyword in normalized:
                query_objects.append(keyword)
        entities.query_objects = query_objects
        
        return entities
    
    def _determine_intents(
        self, 
        scores: Dict[IntentType, float],
        entities: ExtractedEntities
    ) -> tuple[IntentType, List[IntentType], float]:
        """Determine primary and secondary intents from scores"""
        
        # Sort by score
        sorted_intents = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Filter out zero scores
        non_zero = [(intent, score) for intent, score in sorted_intents if score > 0]
        
        if not non_zero:
            # No pattern matches - use GENERAL or HOW_TO based on query structure
            return IntentType.GENERAL, [], 0.5
        
        primary_intent, primary_score = non_zero[0]
        
        # Get secondary intents (score > 0.3 and within 0.3 of primary)
        secondaries = []
        for intent, score in non_zero[1:3]:  # Max 2 secondary
            if score > 0.3 and (primary_score - score) < 0.3:
                secondaries.append(intent)
        
        # Boost confidence based on entity extraction
        confidence = primary_score
        if entities.product_model:
            confidence = min(1.0, confidence + 0.1)
        if entities.error_code:
            confidence = min(1.0, confidence + 0.1)
        
        return primary_intent, secondaries, confidence
    
    def _adjust_by_entities(
        self,
        primary: IntentType,
        secondaries: List[IntentType],
        entities: ExtractedEntities
    ) -> tuple[IntentType, List[IntentType]]:
        """Adjust intent classification based on extracted entities"""
        
        # If error code detected, prioritize ERROR_CODE
        if entities.error_code:
            if primary != IntentType.ERROR_CODE:
                if primary not in secondaries:
                    secondaries = [primary] + secondaries[:1]
                primary = IntentType.ERROR_CODE
        
        # If parameter + value detected, likely CONFIGURATION
        if entities.parameter_type and entities.target_value:
            if primary not in [IntentType.CONFIGURATION, IntentType.SPECIFICATION]:
                if IntentType.CONFIGURATION not in secondaries:
                    secondaries.append(IntentType.CONFIGURATION)
        
        # If accessory mentioned, likely ACCESSORY_QUERY
        if entities.accessory_type:
            if primary != IntentType.ACCESSORY_QUERY:
                if IntentType.ACCESSORY_QUERY not in secondaries:
                    secondaries.append(IntentType.ACCESSORY_QUERY)
        
        # Limit secondaries to 2
        secondaries = secondaries[:2]
        
        return primary, secondaries
    
    def get_intent_info(self, intent: IntentType) -> Dict[str, Any]:
        """Get metadata about an intent type"""
        intent_info = {
            IntentType.ERROR_CODE: {
                "description": "Error code lookup and diagnosis",
                "requires_entities": ["error_code"],
                "document_types": ["error_code_list", "service_bulletin"],
            },
            IntentType.TROUBLESHOOT: {
                "description": "Problem diagnosis and troubleshooting",
                "requires_entities": ["product_model"],
                "document_types": ["service_bulletin", "troubleshooting_guide"],
            },
            IntentType.CONFIGURATION: {
                "description": "Parameter configuration and settings",
                "requires_entities": ["product_model"],
                "document_types": ["configuration_guide", "product_manual"],
            },
            IntentType.COMPATIBILITY: {
                "description": "Tool-controller-accessory compatibility",
                "requires_entities": ["product_model"],
                "document_types": ["compatibility_matrix", "release_notes"],
            },
            IntentType.SPECIFICATION: {
                "description": "Technical specifications lookup",
                "requires_entities": ["product_model"],
                "document_types": ["spec_sheet", "product_manual"],
            },
            IntentType.PROCEDURE: {
                "description": "Step-by-step procedures",
                "requires_entities": [],
                "document_types": ["procedure_guide", "installation_manual"],
            },
            IntentType.CALIBRATION: {
                "description": "Calibration procedures and validation",
                "requires_entities": ["product_model"],
                "document_types": ["calibration_guide", "procedure_guide"],
            },
            IntentType.FIRMWARE: {
                "description": "Firmware update/downgrade procedures",
                "requires_entities": ["product_model"],
                "document_types": ["firmware_guide", "release_notes"],
            },
            IntentType.INSTALLATION: {
                "description": "Initial setup and installation",
                "requires_entities": ["product_model"],
                "document_types": ["installation_manual", "quick_start_guide"],
            },
            IntentType.MAINTENANCE: {
                "description": "Maintenance and preventive care",
                "requires_entities": ["product_model"],
                "document_types": ["maintenance_guide", "product_manual"],
            },
            IntentType.HOW_TO: {
                "description": "General how-to questions",
                "requires_entities": [],
                "document_types": ["product_manual", "procedure_guide"],
            },
            IntentType.COMPARISON: {
                "description": "Product/feature comparison",
                "requires_entities": [],
                "document_types": ["spec_sheet", "product_manual"],
            },
            IntentType.CAPABILITY_QUERY: {
                "description": "Feature and capability queries",
                "requires_entities": ["product_model"],
                "document_types": ["spec_sheet", "product_manual"],
            },
            IntentType.ACCESSORY_QUERY: {
                "description": "Accessory compatibility and info",
                "requires_entities": ["product_model"],
                "document_types": ["accessory_list", "compatibility_matrix"],
            },
            IntentType.GREETING: {
                "description": "User greeting",
                "requires_entities": [],
                "document_types": [],
            },
            IntentType.OFF_TOPIC: {
                "description": "Off-topic question",
                "requires_entities": [],
                "document_types": [],
            },
            IntentType.GENERAL: {
                "description": "General question",
                "requires_entities": [],
                "document_types": ["product_manual"],
            },
            IntentType.CLARIFICATION: {
                "description": "User needs clarification",
                "requires_entities": [],
                "document_types": [],
            },
        }
        
        return intent_info.get(intent, {
            "description": "Unknown intent",
            "requires_entities": [],
            "document_types": [],
        })
