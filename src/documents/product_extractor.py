"""
Product Extractor Module
Identifies Desoutter product families from filenames and text content.
"""
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Productinfo:
    product_line: str  # e.g., "EPB", "CVI3"
    category: str      # e.g., "BATTERY_TOOL", "CONTROLLER"
    confidence: float  # 0.0 to 1.0

class ProductExtractor:
    """extracts product information from document metadata and content"""
    
    # Product Family Definitions
    # Maps product line codes to their categories
    PRODUCT_FAMILIES = {
        # Battery Tools
        "EPB": "BATTERY_TOOL",
        "EPBC": "BATTERY_TOOL", 
        "EABC": "BATTERY_TOOL",
        "EABS": "BATTERY_TOOL",
        "EAB": "BATTERY_TOOL", 
        "BLRT": "BATTERY_TOOL",
        "ELC": "BATTERY_TOOL",
        "XPB": "BATTERY_TOOL",
        "ELS": "BATTERY_TOOL",
        "ELB": "BATTERY_TOOL",
        
        # Corded/Electric Tools
        "EAD": "CORDED_TOOL",
        "EPD": "CORDED_TOOL", 
        "EFD": "CORDED_TOOL",
        "EIDS": "CORDED_TOOL",
        "ERS": "CORDED_TOOL", 
        "ECS": "CORDED_TOOL", 
        "MC": "CORDED_TOOL",
        "EM": "CORDED_TOOL",
        "ERAL": "CORDED_TOOL", 
        "EME": "CORDED_TOOL",
        "EMEL": "CORDED_TOOL",
        
        # Controllers
        "CVI3": "CONTROLLER",
        "CVIC": "CONTROLLER",
        "CVIL": "CONTROLLER", 
        "CVIR": "CONTROLLER",
        "CVIXS": "CONTROLLER",
        "CVI2": "CONTROLLER",
        
        # Ecosystem
        "CONNECT": "ECOSYSTEM",  # Connect-W, Connect-X
        "C-W": "ECOSYSTEM",      # Connect-W short
        "C-X": "ECOSYSTEM",      # Connect-X short
        "SMART HUB": "ECOSYSTEM"
    }

    def __init__(self):
        self._compile_patterns()
        logger.info("ProductExtractor initialized")

    def _compile_patterns(self):
        """Compile regex patterns for performance"""
        self.patterns = {}
        for family in self.PRODUCT_FAMILIES.keys():
            # Matches whole word of product family, case insensitive
            # Special handling for hyphens etc if needed
            if family in ["CONNECT", "SMART HUB"]:
                 pattern =  rf"\b{re.escape(family)}\w*\b" # Match CONNECT-W etc.
            else:
                 pattern = rf"\b{re.escape(family)}\d*[a-zA-Z]*\b" # Match EPB, EPB8, EPBC
            
            self.patterns[family] = re.compile(pattern, re.IGNORECASE)

    def extract_from_filename(self, filename: str) -> List[str]:
        """
        Identify products from filename.
        High confidence source.
        """
        found_products = set()
        # Normalize separators to spaces for regex word boundary compatibility
        # EPB_Troubleshooting -> EPB Troubleshooting
        filename_clean = filename.upper().replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Check against mapped families
        for family, pattern in self.patterns.items():
            if pattern.search(filename_clean):
                found_products.add(family)
        
        # Special case: E-LIT (already handled by replace '-' -> ' ')
        # But we check for "ELIT" specifically just in case
        if "ELIT" in filename_clean.split():
             found_products.add("E-LIT")
        # If "E LIT" or "E-LIT" was present, the pattern loop might miss it if we only look for key
        # We need to ensure E-LIT is captured.
        if "E LIT" in filename_clean or "E-LIT" in filename.upper(): 
             found_products.add("E-LIT")

        return list(found_products)

    def extract_from_content(self, text: str, max_chars: int = 1000) -> List[str]:
        """
        Identify products from first N chars of content.
        Medium confidence source.
        """
        found_products = set()
        # Normalize separators
        text_sample = text[:max_chars].upper().replace('_', ' ').replace('-', ' ')
        
        for family, pattern in self.patterns.items():
            if pattern.search(text_sample):
                found_products.add(family)
                
        if "E LIT" in text_sample or "ELIT" in text_sample:
             found_products.add("E-LIT")

        return list(found_products)

    def get_product_categories(self, filename: str, text: str = "") -> str:
        """
        Get logic-combined product categories for metadata.
        Returns a comma-separated string of unique categories found.
        """
        # 1. Try Filename (Primary)
        products = self.extract_from_filename(filename)
        
        # 2. If no products in filename, try Content
        if not products and text:
            products = self.extract_from_content(text)
            
        # 3. Map to categories
        categories = set()
        for p in products:
            # Handle E-LIT special case
            if p == "E-LIT":
                categories.add("BATTERY_TOOL")
                continue
                
            cat = self.PRODUCT_FAMILIES.get(p)
            if cat:
                categories.add(cat)
                
        # 4. Add the specific product codes themselves too for granular filter
        # e.g. "BATTERY_TOOL, EPB"
        for p in products:
            categories.add(p)
            
        return ", ".join(sorted(list(categories)))
