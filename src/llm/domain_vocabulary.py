"""
=============================================================================
Centralized Domain Vocabulary Module
=============================================================================
Phase 1.2 Refactor: Consolidate query expansion synonyms

This module centralizes all Desoutter-specific technical terminology that was
previously duplicated across:
- src/llm/hybrid_search.py (QueryExpander.DOMAIN_SYNONYMS)
- src/llm/domain_embeddings.py (DomainVocabulary)

Single source of truth for:
- Domain-specific synonyms
- Error codes and messages
- Product series and models
- Components and parts
- Technical specifications
=============================================================================
"""

import re
from typing import Dict, List, Set, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DomainVocabulary:
    """
    Centralized Desoutter-specific technical terminology.
    
    Provides unified vocabulary for:
    - Query expansion (hybrid search)
    - Domain embeddings
    - Entity extraction
    - Synonym matching
    """
    
    # -------------------------------------------------------------------------
    # DOMAIN SYNONYMS (Consolidated from QueryExpander + DomainVocabulary)
    # -------------------------------------------------------------------------
    DOMAIN_SYNONYMS = {
        # Motor/Drive terms
        "motor": ["motor", "spindle", "drive", "actuator", "rotation", "engine", "rotor", "stator"],
        "spindle": ["spindle", "motor", "shaft", "axis", "fixtured spindle", "transducerized spindle"],
        
        # Noise/Vibration terms
        "noise": ["noise", "squealing", "grinding", "humming", "clicking", "sound"],
        "grinding": ["grinding", "noise", "squealing", "friction"],
        "vibration": ["vibration", "shaking", "tremor", "oscillation"],
        
        # Heat terms
        "heat": ["heat", "overheating", "temperature", "thermal", "hot"],
        "overheating": ["overheating", "heat", "thermal", "temperature rise"],
        
        # Mechanical terms
        "bearing": ["bearing", "ball bearing", "bushing", "roller"],
        "gear": ["gear", "gearbox", "transmission", "cog", "planetary gear", "reduction"],
        "leak": ["leak", "leakage", "seepage", "drip"],
        
        # Electrical terms
        "battery": ["battery", "power", "cell", "charge", "battery pack", "charger"],
        "connection": ["connection", "cable", "wire", "connector", "contact"],
        "voltage": ["voltage", "power", "electric", "current", "volt", "V", "vdc", "vac"],
        "current": ["current", "amp", "ampere", "mA", "milliamp"],
        
        # Error terms
        "error": ["error", "fault", "failure", "problem", "issue", "defect"],
        "fault": ["fault", "error", "failure", "malfunction"],
        "failure": ["failure", "error", "fault", "breakdown"],
        
        # Calibration terms
        "calibration": ["calibration", "calibrate", "adjustment", "tuning", "alignment"],
        "torque": ["torque", "tightening", "tension", "nm", "ft-lb"],
        
        # Tool-specific terms
        "wrench": ["wrench", "tool", "torque wrench", "nutrunner", "nut runner"],
        "controller": ["controller", "cvi3", "cvi2", "cvic", "unit", "control box", "ECU"],
        "screwdriver": ["screwdriver", "electric screwdriver", "cordless screwdriver"],
        "drill": ["drill", "drilling tool", "driller"],
        
        # Tool types
        "nutrunner": ["nutrunner", "nut runner", "torque tool", "tightening tool"],
        "angle_head": ["angle head", "angle tool", "right angle"],
        "pistol_grip": ["pistol grip", "pistol tool", "handheld"],
        "straight": ["straight", "inline", "straight tool"],
        "crowfoot": ["crowfoot", "crow foot", "crow-foot"],
        
        # Components
        "transducer": ["transducer", "torque transducer", "strain gauge", "torque sensor"],
        "encoder": ["encoder", "resolver", "angle encoder", "rotation sensor"],
    }
    
    # -------------------------------------------------------------------------
    # DESOUTTER PRODUCT SERIES
    # -------------------------------------------------------------------------
    PRODUCT_SERIES = {
        # Battery Tightening
        "EBP": "Electric Battery Pistol grip nutrunner",
        "EBA": "Electric Battery Angle nutrunner",
        "EBS": "Electric Battery Straight nutrunner",
        "EBSL": "Electric Battery Screwdriver Long",
        "ENB": "Electric Network Battery nutrunner",
        "eBC": "Electric Battery Crowfoot",
        "eBP": "Electric Battery Pistol (industrial)",
        "EPBC": "Electric Pistol Battery Crowfoot",
        "EPB": "Electric Pistol Battery",
        "EAB": "Electric Angle Battery",
        "ExB": "Battery tool",
        "ExD": "Direct cable tool",
        "EABC": "Battery clutch tool",
        
        # Cable Tightening
        "ECP": "Electric Cable Pistol grip nutrunner",
        "ECA": "Electric Cable Angle nutrunner",
        "ECT": "Electric Cable Telescopic nutrunner",
        "ECSF": "Electric Cable Screwdriver Fixtured",
        "ERT": "Electric Reversible Torque tool",
        "EP": "Electric Pistol",
        "ESL": "Electric Screwdriver Long",
        "ELRT": "Electric Low Reaction Torque",
        
        # Fixtured/Robotic
        "EFA": "Electric Fixtured Angle spindle",
        "EFD": "Electric Fixtured Direct nutrunner",
        "EFM": "Electric Fixtured Multi nutrunner",
        "ERF": "Electric Robotic Fixtured spindle",
        "EFMA": "Electric Fixtured Multi Angle",
        "EFBCI": "Electric Fast-integration Basic Inline",
        "EFBCIT": "Electric Fast-integration Basic Inline Telescopic",
        "EFBCA": "Electric Fast-integration Basic Angle",
        
        # Controllers
        "CVI3": "Controller Versatile Interface 3",
        "CVIC": "Controller Versatile Interface Compact",
        "CVIR": "Controller Versatile Interface Robotic",
        "CVI2": "Controller Versatile Interface 2",
        "CONNECT": "Desoutter CONNECT system",
    }
    
    # -------------------------------------------------------------------------
    # ERROR CODES AND MESSAGES
    # -------------------------------------------------------------------------
    ERROR_CODES = {
        # Torque errors
        "E01": "Torque low",
        "E02": "Torque high",
        "E03": "Angle low",
        "E04": "Angle high",
        "E05": "Time low",
        "E06": "Time high",
        "E07": "Rundown torque exceeded",
        "E08": "Rundown angle exceeded",
        
        # Communication errors
        "E10": "Communication timeout",
        "E11": "Protocol error",
        "E12": "CRC error",
        "E13": "Bus error",
        
        # Hardware errors
        "E20": "Motor overcurrent",
        "E21": "Motor overtemperature",
        "E22": "Encoder error",
        "E23": "Battery low",
        "E24": "Battery critical",
        "E25": "Memory error",
        
        # Calibration errors
        "E30": "Calibration expired",
        "E31": "Transducer error",
        "E32": "Zero offset exceeded",
        "E33": "Calibration mismatch",
        
        # Safety errors
        "E40": "Emergency stop active",
        "E41": "Safety circuit open",
        "E42": "Interlock active",
        
        # General errors
        "E50": "General fault",
        "E51": "Configuration error",
        "E52": "Parameter out of range",
        "E99": "Unknown error",
    }
    
    # -------------------------------------------------------------------------
    # ERROR CODE PATTERNS (for normalization)
    # -------------------------------------------------------------------------
    ERROR_PATTERNS = [
        (r'e\s*0*(\d{2,3})', r'E\1'),  # e47 -> E47
        (r'i\s*0*(\d{2,3})', r'I\1'),  # i205 -> I205
        (r'w\s*0*(\d{2,3})', r'W\1'),  # w201 -> W201
    ]
    
    # -------------------------------------------------------------------------
    # COMPONENTS AND PARTS
    # -------------------------------------------------------------------------
    COMPONENTS = {
        "motor": ["motor", "rotor", "stator", "armature", "commutator", "brush", "brushless"],
        "gearbox": ["gearbox", "gear", "planetary gear", "reduction", "gear ratio", "gear train"],
        "transducer": ["transducer", "torque transducer", "strain gauge", "torque sensor"],
        "encoder": ["encoder", "resolver", "angle encoder", "rotation sensor"],
        "battery": ["battery", "battery pack", "cell", "charger", "charging"],
        "controller": ["controller", "control unit", "ECU", "driver", "inverter"],
        "cable": ["cable", "wire", "harness", "connector", "plug"],
        "housing": ["housing", "case", "body", "shell", "enclosure"],
        "trigger": ["trigger", "switch", "button", "lever", "actuator"],
        "led": ["LED", "indicator", "light", "display", "status light"],
    }
    
    # -------------------------------------------------------------------------
    # TECHNICAL SPECIFICATIONS
    # -------------------------------------------------------------------------
    SPECIFICATIONS = {
        "torque": ["torque", "nm", "newton-meter", "ft-lb", "foot-pound", "kgf-cm"],
        "speed": ["speed", "rpm", "rev/min", "revolutions", "rotation speed"],
        "current": ["current", "amp", "ampere", "mA", "milliamp"],
        "voltage": ["voltage", "volt", "V", "vdc", "vac"],
        "power": ["power", "watt", "W", "kW"],
        "weight": ["weight", "kg", "kilogram", "lb", "pound", "mass"],
        "angle": ["angle", "degree", "deg", "rotation", "turn"],
        "time": ["time", "second", "sec", "ms", "millisecond", "duration"],
    }
    
    # -------------------------------------------------------------------------
    # FAULT SYMPTOMS
    # -------------------------------------------------------------------------
    SYMPTOMS = {
        "overheating": ["error E21", "error E20"],
        "no_power": ["error E23", "error E24"],
        "calibration_issue": ["error E30", "error E31", "error E32", "error E33"],
        "communication_failure": ["error E10", "error E11", "error E12", "error E13"],
        "torque_issue": ["error E01", "error E02", "error E07"],
        "angle_issue": ["error E03", "error E04", "error E08"],
    }
    
    # -------------------------------------------------------------------------
    # REPAIR PROCEDURES AND ACTIONS
    # -------------------------------------------------------------------------
    PROCEDURES = {
        "replace": ["replace", "replacement", "swap", "change", "install new"],
        "calibrate": ["calibrate", "calibration", "adjust", "tune", "alignment"],
        "clean": ["clean", "cleaning", "wipe", "remove debris"],
        "inspect": ["inspect", "check", "examine", "verify", "test"],
        "tighten": ["tighten", "secure", "fasten", "torque"],
        "lubricate": ["lubricate", "oil", "grease", "apply lubricant"],
        "reset": ["reset", "reboot", "restart", "power cycle"],
        "update": ["update", "upgrade", "flash", "install firmware"],
        "measure": ["measure", "test", "check voltage", "check current"],
        "disassemble": ["disassemble", "remove", "take apart", "open"],
    }
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    @classmethod
    def get_all_terms(cls) -> Set[str]:
        """Get all domain terms as a flat set."""
        terms = set()
        
        # Add all synonym keys and values
        for key, synonyms in cls.DOMAIN_SYNONYMS.items():
            terms.add(key)
            terms.update(synonyms)
        
        # Add product series
        terms.update(cls.PRODUCT_SERIES.keys())
        
        # Add error codes
        terms.update(cls.ERROR_CODES.keys())
        
        # Add component terms
        for component_list in cls.COMPONENTS.values():
            terms.update(component_list)
        
        return terms
    
    @classmethod
    def get_synonyms(cls, term: str) -> List[str]:
        """
        Get synonyms for a domain term.
        
        Args:
            term: Term to find synonyms for
            
        Returns:
            List of synonyms (empty if term not found)
        """
        term_lower = term.lower()
        
        # Check DOMAIN_SYNONYMS
        if term_lower in cls.DOMAIN_SYNONYMS:
            return cls.DOMAIN_SYNONYMS[term_lower]
        
        # Check if term is in any synonym list
        for key, synonyms in cls.DOMAIN_SYNONYMS.items():
            if term_lower in [s.lower() for s in synonyms]:
                return cls.DOMAIN_SYNONYMS[key]
        
        return []
    
    @classmethod
    def get_related_errors(cls, symptom: str) -> List[str]:
        """
        Get error codes related to a symptom.
        
        Args:
            symptom: Symptom keyword
            
        Returns:
            List of related error codes
        """
        symptom_lower = symptom.lower()
        
        if symptom_lower in cls.SYMPTOMS:
            return cls.SYMPTOMS[symptom_lower]
        
        return []
    
    @classmethod
    def normalize_error_code(cls, query: str) -> str:
        """
        Normalize error code formats (e47 -> E047).
        
        Args:
            query: Query string potentially containing error codes
            
        Returns:
            Query with normalized error codes
        """
        result = query
        for pattern, replacement in cls.ERROR_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    @classmethod
    def expand_product_abbreviation(cls, abbrev: str) -> Optional[str]:
        """
        Expand product abbreviation to full name.
        
        Args:
            abbrev: Product abbreviation (e.g., "CVI3")
            
        Returns:
            Full product name or None if not found
        """
        return cls.PRODUCT_SERIES.get(abbrev.upper())
    
    @classmethod
    def get_error_message(cls, error_code: str) -> Optional[str]:
        """
        Get error message for an error code.
        
        Args:
            error_code: Error code (e.g., "E01")
            
        Returns:
            Error message or None if not found
        """
        return cls.ERROR_CODES.get(error_code.upper())


# Singleton instance for easy access
_vocabulary = DomainVocabulary()


def get_domain_vocabulary() -> DomainVocabulary:
    """Get singleton DomainVocabulary instance."""
    return _vocabulary


# Module-level logger
logger.info("✅ Centralized DomainVocabulary loaded (Phase 1.2 refactor)")
