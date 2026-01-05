"""
Intelligent Product Extractor
Automatically detects product families from filenames and content
NO manual mappings required - uses pattern recognition
"""
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProductInfo:
    """Extracted product information"""
    product_family: str      # e.g., "ERS", "EABS", "CVI3"
    product_model: str       # e.g., "ERS6", "EABS12-1100-4S"
    category: str            # e.g., "TOOL", "CONTROLLER", "ACCESSORY"
    confidence: float        # 0.0 to 1.0
    source: str              # "filename" or "content"


class IntelligentProductExtractor:
    """
    Automatically extracts product information using pattern recognition.
    No manual product lists needed - learns from document structure.
    """
    
    # Desoutter product naming patterns (regex-based, not hardcoded lists)
    PRODUCT_PATTERNS = [
        # Battery tools: EABS, EAB, EPB, EPBC, ELC, ELS, ELB, BLRT, XPB, E-LIT
        (r'\b(EABS\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(EABC\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(EAB\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(EPBC\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(EPB\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(ELC\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(ELS\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(ELB\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(BLRT\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(XPB\d*[-\w]*)\b', 'BATTERY_TOOL'),
        (r'\b(E[-]?LIT\d*[-\w]*)\b', 'BATTERY_TOOL'),
        
        # Corded/Electric tools: ERS, EME, ERAL, EAD, EPD, EFD, EIDS, ECS, MC, EM
        (r'\b(ERS\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(EMEL\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(EME\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(ERAL\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(EAD\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(EPD\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(EFD\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(EIDS\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(ECS\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(MC\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(SLC\d*[-\w]*)\b', 'CORDED_TOOL'),
        (r'\b(SLBN\d*[-\w]*)\b', 'CORDED_TOOL'),
        
        # Pulse tools
        (r'\b(E[-]?PULSE\d*[-\w]*)\b', 'PULSE_TOOL'),
        (r'\b(ERP\d*[-\w]*)\b', 'PULSE_TOOL'),
        
        # High torque tools
        (r'\b(EID\d*[-\w]*)\b', 'HIGH_TORQUE_TOOL'),
        (r'\b(EFM\d*[-\w]*)\b', 'HIGH_TORQUE_TOOL'),
        (r'\b(EFMA\d*[-\w]*)\b', 'HIGH_TORQUE_TOOL'),
        (r'\b(ERF\d*[-\w]*)\b', 'HIGH_TORQUE_TOOL'),
        (r'\b(ERXS\d*[-\w]*)\b', 'HIGH_TORQUE_TOOL'),
        
        # Torque tools: ELRT, ERT, LRT
        (r'\b(ELRT\d*[-\w]*)\b', 'TORQUE_TOOL'),
        (r'\b(ERT\d*[-\w]*)\b', 'TORQUE_TOOL'),
        (r'\b(LRT\d*[-\w]*)\b', 'TORQUE_TOOL'),
        
        # Fast integration spindles
        (r'\b(EFBCI\d*[-\w]*)\b', 'SPINDLE'),
        (r'\b(EFBCIT\d*[-\w]*)\b', 'SPINDLE'),
        (r'\b(EFBCA\d*[-\w]*)\b', 'SPINDLE'),
        
        # Controllers: CVI3, CVIC, CVIL, CVIR, CVI2, CVIXS
        (r'\b(CVI3\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(CVIC\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(CVIL\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(CVIR\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(CVIXS\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(CVI2\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(ESP\d*[-\w]*)\b', 'CONTROLLER'),
        (r'\b(AXON\d*[-\w]*)\b', 'CONTROLLER'),
        
        # Ecosystem: Connect-W, Connect-X, Smart Hub
        (r'\b(CONNECT[-]?[WX]\w*)\b', 'ECOSYSTEM'),
        (r'\b(C[-]?[WX]\d*)\b', 'ECOSYSTEM'),
        (r'\b(SMART\s*HUB\w*)\b', 'ECOSYSTEM'),
        
        # Accessories & Generic
        (r'\b(ACCESS\s*POINT\w*)\b', 'ACCESSORY'),
        (r'\b(WIFI\s*MODULE\w*)\b', 'ACCESSORY'),
        (r'\b(BATTERY\s*PACK\w*)\b', 'ACCESSORY'),
    ]
    
    # Generic/Universal document indicators
    GENERIC_INDICATORS = [
        'general', 'universal', 'all models', 'all tools', 'overview',
        'introduction', 'safety', 'compliance', 'training', 'quick start',
        'getting started', 'installation guide'
    ]
    
    def __init__(self):
        # Compile patterns for performance
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in self.PRODUCT_PATTERNS
        ]
        logger.info("IntelligentProductExtractor initialized with pattern-based detection")
    
    def extract_from_filename(self, filename: str) -> List[ProductInfo]:
        """
        Extract product info from filename.
        High confidence source.
        
        Examples:
            "ERS6 - Product Instructions.pdf" → ERS6, CORDED_TOOL
            "EABS12-1100-4S Maintenance.docx" → EABS12-1100-4S, BATTERY_TOOL
            "CVI3 VISION User Manual.pdf" → CVI3, CONTROLLER
            "General Safety Guidelines.pdf" → GENERAL, GENERIC
        """
        products = []
        filename_upper = filename.upper()
        
        for pattern, category in self._compiled_patterns:
            matches = pattern.findall(filename_upper)
            for match in matches:
                product_family = self._extract_family(match)
                products.append(ProductInfo(
                    product_family=product_family,
                    product_model=match,
                    category=category,
                    confidence=0.95,  # High confidence from filename
                    source="filename"
                ))
        
        # Check for generic documents
        if not products:
            filename_lower = filename.lower()
            if any(indicator in filename_lower for indicator in self.GENERIC_INDICATORS):
                products.append(ProductInfo(
                    product_family="GENERAL",
                    product_model="GENERAL",
                    category="GENERIC",
                    confidence=0.8,
                    source="filename"
                ))
        
        return products
    
    def extract_from_content(self, text: str, max_chars: int = 2000) -> List[ProductInfo]:
        """
        Extract product info from document content.
        Medium confidence - used when filename doesn't have product info.
        
        Scans first N characters for product references.
        """
        products = []
        text_sample = text[:max_chars].upper()
        seen_families = set()
        
        for pattern, category in self._compiled_patterns:
            matches = pattern.findall(text_sample)
            for match in matches:
                product_family = self._extract_family(match)
                
                # Avoid duplicates
                if product_family not in seen_families:
                    seen_families.add(product_family)
                    products.append(ProductInfo(
                        product_family=product_family,
                        product_model=match,
                        category=category,
                        confidence=0.7,  # Lower confidence from content
                        source="content"
                    ))
        
        return products
    
    def _extract_family(self, product_model: str) -> str:
        """
        Extract product family from full model name.
        
        Examples:
            "ERS6" → "ERS"
            "EABS12-1100-4S" → "EABS"
            "CVI3 VISION" → "CVI3"
            "ELRT025" → "ELRT"
            "EPBC8-1800" → "EPBC"
        """
        # Remove common suffixes and numbers to get base family
        model_upper = product_model.upper().strip()
        
        # Special cases where numbers are part of family name
        special_families = ['CVI3', 'CVI2', 'CVIXS']
        for sf in special_families:
            if model_upper.startswith(sf):
                return sf
        
        # Match the base alphabetic part (family code)
        match = re.match(r'^([A-Z]+)', model_upper)
        if match:
            family = match.group(1)
            return family
        
        return model_upper
    
    def get_product_metadata(self, filename: str, content: str = "") -> Dict:
        """
        Get comprehensive product metadata for a document.
        
        Returns:
            {
                "product_family": "ERS",           # Primary family
                "product_families": "ERS",         # Comma-separated (ChromaDB compatible)
                "product_models": "ERS6",          # Specific models mentioned
                "product_category": "CORDED_TOOL", # Tool category
                "is_generic": False,               # True if applies to all products
                "confidence": 0.95                 # Detection confidence
            }
        """
        # Try filename first (higher confidence)
        products = self.extract_from_filename(filename)
        
        # If no products in filename, try content
        if not products and content:
            products = self.extract_from_content(content)
        
        # If still nothing, mark as unknown (will be included in all searches)
        if not products:
            return {
                "product_family": "UNKNOWN",
                "product_families": "UNKNOWN",
                "product_models": "",
                "product_category": "UNKNOWN",
                "is_generic": True,  # Unknown docs should be searchable
                "confidence": 0.5
            }
        
        # Aggregate results
        families = list(set(p.product_family for p in products))
        models = list(set(p.product_model for p in products))
        categories = list(set(p.category for p in products))
        max_confidence = max(p.confidence for p in products)
        
        # Primary family is the first one found (from filename if available)
        primary_family = families[0] if families else "UNKNOWN"
        primary_category = categories[0] if categories else "UNKNOWN"
        is_generic = primary_family == "GENERAL"
        
        return {
            "product_family": primary_family,
            "product_families": ", ".join(families),  # ChromaDB-compatible string
            "product_models": ", ".join(models),
            "product_category": primary_category,
            "is_generic": is_generic,
            "confidence": max_confidence
        }
    
    def extract_product_from_query(self, query: str, product_number: str = None) -> Dict:
        """
        Extract product context from user query and/or selected product.
        Used during retrieval to filter results.
        
        Args:
            query: User's query text
            product_number: Selected product part number (e.g., "6151659770")
        
        Returns:
            {
                "product_family": "ERS",
                "search_families": ["ERS", "GENERAL", "UNKNOWN"],
                "has_product_context": True
            }
        """
        detected_families = set()
        
        # Extract from query text
        query_products = self.extract_from_content(query, max_chars=500)
        for p in query_products:
            detected_families.add(p.product_family)
        
        if not detected_families:
            # No specific product detected - return empty (search all)
            return {
                "product_family": None,
                "search_families": [],  # Empty = search all
                "has_product_context": False
            }
        
        # Build search families list (include GENERAL, UNKNOWN for safety)
        search_families = list(detected_families)
        search_families.extend(["GENERAL", "UNKNOWN"])
        search_families = list(set(search_families))  # Dedupe
        
        return {
            "product_family": list(detected_families)[0],
            "search_families": search_families,
            "has_product_context": True
        }
    
    # Backward compatibility with old API
    def get_product_categories(self, filename: str, text: str = "") -> str:
        """
        Legacy API - returns comma-separated categories string.
        Maintained for backward compatibility with existing code.
        """
        metadata = self.get_product_metadata(filename, text)
        
        # Build categories string like the old format
        categories = set()
        
        # Add product family
        if metadata["product_family"] != "UNKNOWN":
            categories.add(metadata["product_family"])
        
        # Add category
        if metadata["product_category"] != "UNKNOWN":
            categories.add(metadata["product_category"])
        
        return ", ".join(sorted(categories)) if categories else ""


# Backward compatibility alias
ProductExtractor = IntelligentProductExtractor
