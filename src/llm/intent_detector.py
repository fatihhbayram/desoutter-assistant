"""
Query Intent Detection Module
=============================
Classifies user queries into specific intent categories to enable
context-appropriate response generation.

Intent Types:
- ERROR_CODE: E01, E02, error code lookup (highest priority)
- TROUBLESHOOTING: Error, fault, problem diagnosis
- SPECIFICATIONS: Torque, speed, dimensions, technical specs
- CALIBRATION: Calibrate, adjust, zero setting
- MAINTENANCE: Clean, lubricate, replace parts
- CONNECTION: WiFi, ethernet, network, pairing
- INSTALLATION: Setup, assembly, configuration
- GENERAL: Default for unclear intent

Version: 2.0 - Simplified keyword-based matching for reliability
"""
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """
    Query intent classification - El-Harezmi Architecture
    
    Expanded from 8 to 15 intent types for better response targeting.
    
    Original (8):
        TROUBLESHOOTING, SPECIFICATIONS, INSTALLATION, CALIBRATION,
        MAINTENANCE, CONNECTION, ERROR_CODE, GENERAL
    
    New (7):
        CONFIGURATION, COMPATIBILITY, PROCEDURE, FIRMWARE,
        COMPARISON, CAPABILITY_QUERY, ACCESSORY_QUERY
    """
    # Original intents (8)
    TROUBLESHOOTING = "troubleshooting"   # Error, fault, problem diagnosis
    SPECIFICATIONS = "specifications"      # Torque, speed, weight, dimensions
    INSTALLATION = "installation"          # Setup, assembly, first-time config
    CALIBRATION = "calibration"            # Calibrate, adjust, zero setting
    MAINTENANCE = "maintenance"            # Clean, lubricate, replace parts
    CONNECTION = "connection"              # WiFi, ethernet, network, pairing
    ERROR_CODE = "error_code"              # E01, E02, error code lookup
    GENERAL = "general"                    # Default for unclear intent
    
    # New intents (7) - El-Harezmi Phase 1
    CONFIGURATION = "configuration"        # Pset, parameter setup, settings
    COMPATIBILITY = "compatibility"        # Tool-controller compatibility
    PROCEDURE = "procedure"                # Step-by-step instructions
    FIRMWARE = "firmware"                  # Firmware update/downgrade
    COMPARISON = "comparison"              # Model comparison, "hangisi daha iyi"
    CAPABILITY_QUERY = "capability"        # "WiFi var mı?", "Max tork nedir?"
    ACCESSORY_QUERY = "accessory"          # Battery, dock, adapter questions


@dataclass
class IntentResult:
    """Result of intent detection"""
    intent: QueryIntent  # Primary detected intent
    confidence: float  # 0.0-1.0 confidence score
    matched_patterns: List[str]  # Patterns that matched
    secondary_intent: Optional[QueryIntent] = None  # Secondary intent if applicable


class IntentDetector:
    """
    Simple, reliable keyword-based intent detection.
    
    Uses pattern matching to classify query intent for proper routing
    and response generation.
    
    Priority Order (highest to lowest):
    1. error_code - Most specific (regex patterns)
    2. troubleshooting - Problem diagnosis
    3. specifications - Technical data requests
    4. calibration - Adjustment procedures
    5. maintenance - Service procedures
    6. connection - Network/connectivity
    7. installation - Setup procedures
    8. general - Default fallback
    """
    
    def __init__(self):
        """Initialize intent detector with keyword patterns"""
        
        # Error code patterns (highest priority - regex based)
        self.error_patterns = [
            r'\be\d{2,4}\b',           # E01, E804, E4502
            r'error\s*code',            # "error code" or "errorcode"
            r'error\s+\d+',             # "error 123"
            r'fault\s*code',            # "fault code"
            r'fault\s+\d+',             # "fault 123"
            r'hata\s*kodu',             # Turkish: error code
            r'alarm\s+\d+',             # "alarm 47"
        ]
        
        # Intent keyword lists - using simple substring matching
        # Order in dict determines tie-breaker priority
        self.intent_keywords = {
            'troubleshooting': [
                # English - Operation issues
                'stop', 'stops', 'stopped', 'stopping',
                'not working', "won't", "wont", "doesn't", "doesnt",
                'not start', 'not run', 'not turn',
                'broken', 'break', 'broke',
                'fail', 'fails', 'failed', 'failure',
                'problem', 'issue', 'trouble',
                'turn off', 'turns off', 'turned off', 'shut down', 'shuts down',
                'random', 'randomly', 'intermittent', 'sometimes',
                'malfunction', 'defect', 'defective',
                'overheat', 'overheating', 'hot', 'heating',
                'noise', 'noisy', 'grinding', 'vibrat', 'vibration',
                'stuck', 'jam', 'jammed',
                'slow', 'weak', 'low power',
                # Turkish
                'calismiyor', 'calışmıyor', 'çalışmıyor',
                'sorun', 'ariza', 'arıza', 'bozuk',
                'durdu', 'duruyor',
            ],
            
            'specifications': [
                # English
                'torque', 'rpm', 'speed',
                'power', 'watt', 'voltage', 'volt', 'amp', 'amperage',
                'weight', 'dimension', 'size', 'length', 'width', 'height',
                'capacity', 'specification', 'specs', 'spec',
                'what is the', 'what are the',
                'how much', 'how many', 'how heavy', 'how fast',
                'maximum', 'minimum', 'max', 'min',
                'rated', 'nominal', 'range',
                # Turkish
                'tork', 'hiz', 'hız', 'guc', 'güç', 'agirlik', 'ağırlık', 'boyut',
                'ne kadar', 'kac', 'kaç',
            ],
            
            'calibration': [
                # English
                'calibrat', 'calibration', 'calibrate',
                'adjust', 'adjustment', 'adjusting',
                'zero', 'zeroing', 'offset',
                'accuracy', 'accurate',
                'tune', 'tuning',
                'how to calibrate', 'how to adjust',
                # Turkish
                'kalibr', 'kalibrasyon', 'ayarla', 'sifirla', 'sıfırla',
            ],
            
            'maintenance': [
                # English
                'maintain', 'maintenance',
                'service', 'servicing',
                'clean', 'cleaning',
                'lubricat', 'lubrication', 'lubricate', 'oil', 'grease',
                'replace', 'replacement', 'replacing',
                'inspect', 'inspection',
                'schedule', 'interval', 'period', 'periyod',
                'preventive', 'routine', 'regular', 'periodic',
                'how often', 'when should',
                # Turkish
                'bakim', 'bakım', 'servis', 'temizl', 'yagla', 'yağla',
                'bakim periyodu', 'bakım periyodu',
            ],
            
            'connection': [
                # English
                'wifi', 'wi-fi', 'wireless',
                'network', 'ethernet', 'lan',
                'connect', 'connection', 'connectivity', 'disconnect',
                'cable', 'wire', 'wiring',
                'plug', 'socket', 'port',
                'communication', 'signal',
                'pair', 'pairing', 'link',
                'ip address', 'network setting',
                'cannot connect',
                # Turkish
                'baglanti', 'bağlantı', 'kablosuz', 'kablo', 'ag', 'ağ',
            ],
            
            'installation': [
                # English
                'install', 'installation', 'installing',
                'setup', 'set up', 'set-up',
                'mount', 'mounting', 'mounted',
                'assemble', 'assembly', 'assembling',
                'configure', 'configuration', 'configuring',
                'initial', 'initialize', 'initialization',
                'how to install', 'how to setup', 'how to mount',
                'first time', 'getting started',
                # Turkish
                'kurulum', 'montaj', 'tak', 'yerlestir', 'yerleştir',
            ],
            
            # =================================================================
            # NEW INTENTS (El-Harezmi Phase 1)
            # =================================================================
            
            'configuration': [
                # English - Parameter/Pset setup
                'pset', 'parameter', 'setting', 'settings',
                'torque setting', 'angle setting', 'speed setting',
                'how to set', 'how to configure', 'configure parameter',
                'change parameter', 'modify setting', 'adjust setting',
                'target torque', 'target angle', 'set value',
                'program', 'programming', 'batch',
                # Turkish
                'ayar', 'ayarla', 'ayarlama', 'parametre',
                'pset ayarı', 'pset ayar', 'tork ayarı', 'açı ayarı',
                'hız ayarı', 'değer ayarla', 'program ayarı',
            ],
            
            'compatibility': [
                # English - Tool/Controller matching
                'compatible', 'compatibility', 'work with', 'works with',
                'support', 'supports', 'supported',
                'which controller', 'which version', 'what version',
                'can i use', 'is it compatible', 'does it work',
                'require', 'requirement', 'needed for',
                # Turkish
                'uyumlu', 'uyumluluk', 'çalışır mı', 'ile çalışır',
                'hangi controller', 'hangi versiyon', 'hangi sürüm',
                'gerekli mi', 'gereksinim', 'destekliyor mu',
            ],
            
            'procedure': [
                # English - Step-by-step guides
                'step by step', 'step-by-step', 'procedure', 'procedures',
                'instructions', 'instruction', 'guide', 'tutorial',
                'process', 'method', 'way to', 'how do i',
                'walkthrough', 'steps to', 'sequence',
                # Turkish
                'adım adım', 'prosedür', 'nasıl yapılır', 'yapılır mı',
                'talimat', 'rehber', 'yöntem', 'işlem', 'sıra',
            ],
            
            'firmware': [
                # English - Software/Firmware updates
                'firmware', 'update', 'upgrade', 'downgrade',
                'version', 'flash', 'software update', 'software version',
                'latest version', 'new version', 'update firmware',
                # Turkish
                'yazılım', 'güncelleme', 'güncelle', 'versiyon', 'sürüm',
                'yazılım güncelleme', 'firmware güncelleme',
            ],
            
            'comparison': [
                # English - Model comparisons
                'compare', 'comparison', 'difference', 'differences',
                'vs', 'versus', 'or', 'better', 'which one',
                'between', 'pros and cons', 'advantages',
                # Turkish
                'karşılaştır', 'karşılaştırma', 'fark', 'farkı',
                'hangisi', 'arasındaki fark', 'mı yoksa', 'daha iyi',
            ],
            
            'capability': [
                # English - Feature/capability questions
                'does it have', 'can it', 'is there', 'has',
                'feature', 'capability', 'able to', 'capable',
                'support wifi', 'has bluetooth', 'has wifi',
                'maximum', 'minimum', 'limit', 'range',
                # Turkish
                'var mı', 'yapabilir mi', 'özellik', 'özelliği',
                'wifi var mı', 'bluetooth var mı', 'destekler mi',
                'maksimum', 'minimum', 'sınır', 'aralık',
            ],
            
            'accessory': [
                # English - Accessory questions
                'battery', 'batteries', 'charger', 'charging',
                'dock', 'docking', 'adapter', 'cable', 'cables',
                'which battery', 'which charger', 'accessory', 'accessories',
                'spare part', 'replacement part',
                # Turkish
                'batarya', 'pil', 'şarj', 'şarj cihazı', 'şarj aleti',
                'dok', 'adaptör', 'kablo', 'aksesuar',
                'hangi batarya', 'hangi şarj', 'yedek parça',
            ],
        }
        
        logger.info("IntentDetector initialized (v2.0 - keyword-based)")
    
    def detect_intent(
        self,
        query: str,
        product_info: Optional[Dict] = None
    ) -> IntentResult:
        """
        Detect query intent using keyword pattern matching.
        
        Args:
            query: User query text
            product_info: Optional product information for context (not used in v2)
        
        Returns:
            IntentResult with detected intent and confidence
            
        Example:
            >>> detector = IntentDetector()
            >>> result = detector.detect_intent("Motor stops during operation")
            >>> result.intent.value
            'troubleshooting'
        """
        query_lower = query.lower()
        logger.debug(f"[INTENT] Analyzing query: '{query}'")
        
        # =====================================================================
        # Priority 1: Check for error codes (most specific - regex patterns)
        # =====================================================================
        for pattern in self.error_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                logger.info(f"[INTENT] Detected: error_code (matched pattern: {pattern}, found: '{match.group()}')")
                return IntentResult(
                    intent=QueryIntent.ERROR_CODE,
                    confidence=0.95,
                    matched_patterns=[pattern],
                    secondary_intent=None
                )
        
        # =====================================================================
        # Priority 2: Check all other intents by keyword matching
        # =====================================================================
        intent_scores: Dict[str, Dict] = {}
        
        for intent_name, keywords in self.intent_keywords.items():
            matches = []
            for keyword in keywords:
                if keyword in query_lower:
                    matches.append(keyword)
            
            if matches:
                intent_scores[intent_name] = {
                    'count': len(matches),
                    'keywords': matches,
                    # Bonus for longer keyword matches (more specific)
                    'specificity': sum(len(k) for k in matches)
                }
                logger.debug(f"[INTENT] {intent_name}: {len(matches)} matches -> {matches}")
        
        # =====================================================================
        # Determine winner based on match count, then specificity
        # =====================================================================
        if intent_scores:
            # Sort by: 1) match count (desc), 2) specificity (desc)
            best_intent = max(
                intent_scores.keys(),
                key=lambda k: (intent_scores[k]['count'], intent_scores[k]['specificity'])
            )
            match_info = intent_scores[best_intent]
            
            # Calculate confidence based on match count
            confidence = min(0.5 + (match_info['count'] * 0.15), 0.95)
            
            # Find secondary intent
            secondary = None
            remaining = {k: v for k, v in intent_scores.items() if k != best_intent}
            if remaining:
                secondary_name = max(remaining.keys(), key=lambda k: remaining[k]['count'])
                if remaining[secondary_name]['count'] >= 1:
                    secondary = QueryIntent(secondary_name)
            
            logger.info(
                f"[INTENT] Detected: {best_intent} "
                f"(confidence={confidence:.2f}, "
                f"matches={match_info['count']}: {match_info['keywords']})"
            )
            
            return IntentResult(
                intent=QueryIntent(best_intent),
                confidence=confidence,
                matched_patterns=match_info['keywords'],
                secondary_intent=secondary
            )
        
        # =====================================================================
        # Default: GENERAL (no keywords matched)
        # =====================================================================
        logger.info("[INTENT] Detected: general (no specific keywords matched)")
        return IntentResult(
            intent=QueryIntent.GENERAL,
            confidence=0.5,
            matched_patterns=[],
            secondary_intent=None
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
            # Original intents
            QueryIntent.TROUBLESHOOTING: "Diagnose and fix problems",
            QueryIntent.SPECIFICATIONS: "Technical specifications and parameters",
            QueryIntent.INSTALLATION: "Setup and installation procedures",
            QueryIntent.CALIBRATION: "Calibration and adjustment",
            QueryIntent.MAINTENANCE: "Maintenance and service procedures",
            QueryIntent.CONNECTION: "Network and connectivity setup",
            QueryIntent.ERROR_CODE: "Error code lookup and resolution",
            QueryIntent.GENERAL: "General inquiry",
            # New intents (El-Harezmi)
            QueryIntent.CONFIGURATION: "Parameter and Pset configuration",
            QueryIntent.COMPATIBILITY: "Tool-controller compatibility check",
            QueryIntent.PROCEDURE: "Step-by-step procedure guide",
            QueryIntent.FIRMWARE: "Firmware update/downgrade procedures",
            QueryIntent.COMPARISON: "Model comparison and selection",
            QueryIntent.CAPABILITY_QUERY: "Feature and capability inquiry",
            QueryIntent.ACCESSORY_QUERY: "Accessory and spare parts inquiry",
        }
        return descriptions.get(intent, "Unknown intent")


# =============================================================================
# QUICK TEST (run with: python -m src.llm.intent_detector)
# =============================================================================
if __name__ == "__main__":
    detector = IntentDetector()
    
    test_cases = [
        # Original test cases
        ("Motor won't start", "troubleshooting"),
        ("Motor stops during operation", "troubleshooting"),
        ("Tool turns off randomly", "troubleshooting"),
        ("Grinding noise from the tool", "troubleshooting"),
        ("Tool overheating during operation", "troubleshooting"),
        ("What is the maximum torque?", "specifications"),
        ("Error code E804", "error_code"),
        ("E047 hata kodu ne anlama geliyor?", "error_code"),
        ("How to calibrate torque settings?", "calibration"),
        ("Cable connection between tool and CVI3", "connection"),
        ("WiFi connection issues", "connection"),
        ("How to install the tool on workstation?", "installation"),
        ("How often should I lubricate?", "maintenance"),
        ("Bakım periyodu ne kadar?", "maintenance"),
        ("Tell me about this tool", "general"),
        
        # NEW: Configuration intent
        ("EABC-3000 pset ayarı nasıl yapılır?", "configuration"),
        ("How to set torque to 50 Nm?", "configuration"),
        ("Tork ayarı nasıl değiştirilir?", "configuration"),
        ("Change parameter settings", "configuration"),
        
        # NEW: Compatibility intent
        ("EABC-3000 hangi CVI3 ile çalışır?", "compatibility"),
        ("Is this tool compatible with CVI3 v2.5?", "compatibility"),
        ("Which controller works with ERS6?", "compatibility"),
        ("Bu alet CVIR II ile uyumlu mu?", "compatibility"),
        
        # NEW: Procedure intent
        ("Step by step firmware update", "firmware"),  # firmware is primary when mentioned
        ("Kalibrasyon prosedürü adımları", "calibration"),  # calibration is primary
        ("Give me instructions for setup", "procedure"),
        ("How do I perform the installation procedure?", "procedure"),
        
        # NEW: Firmware intent
        ("How to update firmware?", "firmware"),
        ("Firmware güncelleme nasıl yapılır?", "firmware"),
        ("What is the latest firmware version?", "firmware"),
        
        # NEW: Comparison intent
        ("ERS6 vs ERS12 difference", "comparison"),
        ("Hangisi daha iyi EPBC mi EPB mi?", "comparison"),
        ("Compare EABC-3000 and EABC-2500", "comparison"),
        
        # NEW: Capability intent
        ("Does this tool have WiFi feature?", "capability"),
        ("Bu alette bluetooth var mı?", "capability"),
        ("What features does EABC-3000 have?", "capability"),
        
        # NEW: Accessory intent
        ("Which battery works with EPBC?", "accessory"),
        ("Hangi batarya ile çalışır?", "accessory"),
        ("What charger should I use for ELC?", "accessory"),
        ("Hangi şarj cihazı kullanmalıyım?", "accessory"),
    ]
    
    print("=" * 70)
    print("INTENT DETECTOR v2.0 - VALIDATION TEST")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = detector.detect_intent(query)
        detected = result.intent.value
        is_correct = detected == expected
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"
        
        print(f'{status} "{query}"')
        print(f'   Expected: {expected}, Got: {detected} (conf: {result.confidence:.2f})')
        if result.matched_patterns:
            print(f'   Matched: {result.matched_patterns[:3]}')
        print()
    
    print("=" * 70)
    print(f"Results: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.0f}%)")
    print(f"Target: >=90% (>={int(len(test_cases)*0.9)}/{len(test_cases)})")
    print("=" * 70)
