"""
Product Categorization Module for Schema v2
==========================================
Helper functions to detect tool_category, wireless info, platform connection,
modular system, product family, and tool type from scraped product data.

Business Rules:
- Battery tools: Detect WiFi capability from model name ("C" suffix) or description
- Cable tools: Map to compatible platforms (CVI3, CVIR II, etc.)
- Electric drilling: Detect modular system (base tools vs attachments)
"""
import re
from typing import Dict, Optional, Tuple, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# URL patterns for category detection
CATEGORY_URL_PATTERNS = {
    "battery_tightening": "https://www.desouttertools.com/en/c/battery-tightening-tools",
    "cable_tightening": "https://www.desouttertools.com/en/c/cable-tightening-tools",
    "electric_drilling": "https://www.desouttertools.com/en/c/electric-drilling-tools",
    "platform": "https://www.desouttertools.com/en/c/corded-platforms",
}

# Product family patterns (prefix in model name)
PRODUCT_FAMILIES = [
    "EPBC", "EABS", "EIBS",  # Battery with WiFi variants (must be before EPB, EAB)
    "EPB", "EAB", "EID", "EPD", "EAD", "ERP", "ERS", "ECS", "ERXS",  # Standard families
    "SLC", "SLBN",  # Low voltage screwdrivers
    "EFD", "EFM", "ERF", "EFMA", "EFBCI", "EFBCIT", "EFBCA",  # Fixtured spindles
    "XPB",  # Modular drilling
]

# Tool type patterns in series name / model name
TOOL_TYPE_PATTERNS = {
    "pistol": [r"pistol", r"grip"],
    "angle_head": [r"angle\s*head", r"angle"],
    "inline": [r"in-?line", r"straight"],
    "screwdriver": [r"screwdriver", r"screw\s*driver"],
    "drill": [r"drill", r"drilling"],
    "fixtured": [r"fixtured", r"spindle", r"fixed"],
}

# Cable tool family to compatible platforms mapping
CABLE_TOOL_PLATFORMS = {
    # EAD, EID, EPD, ERP series - CVI3 compatible
    "EAD": ["CVI3", "CVI3LT"],
    "EID": ["CVI3", "CVI3LT"],
    "EPD": ["CVI3", "CVI3LT"],
    "ERP": ["CVI3", "CVI3LT"],
    # ERS, ECS series - Multiple platform support
    "ERS": ["CVI3", "CVI3LT", "CVIR II"],
    "ECS": ["CVI3", "CVIR II", "ESP-C"],
    "ERXS": ["CVI3", "CVI3LT"],
    # Low voltage screwdrivers
    "SLC": ["CVI3", "ESP-C"],
    "SLBN": ["ESP-C", "ESP"],
    # Fixtured spindles - CVI3 primarily
    "EFD": ["CVI3"],
    "EFM": ["CVI3"],
    "ERF": ["CVI3"],
    "EFMA": ["CVI3"],
    "EFBCI": ["CVI3"],
    "EFBCIT": ["CVI3"],
    "EFBCA": ["CVI3"],
}

# Modular system base tools
MODULAR_BASE_TOOLS = ["XPB-Modular", "XPB-One"]


# =============================================================================
# CATEGORY DETECTION
# =============================================================================

def detect_tool_category(category_url: str, legacy_category: str = "") -> str:
    """
    Detect tool_category from category URL or legacy category name.
    
    Args:
        category_url: The URL of the category page
        legacy_category: Legacy category string (fallback)
        
    Returns:
        Tool category: battery_tightening | cable_tightening | electric_drilling | platform | unknown
    """
    url_lower = category_url.lower() if category_url else ""
    
    # Check URL patterns
    if "battery-tightening" in url_lower:
        return "battery_tightening"
    elif "cable-tightening" in url_lower:
        return "cable_tightening"
    elif "electric-drilling" in url_lower or "drilling-tools" in url_lower:
        return "electric_drilling"
    elif "corded-platforms" in url_lower or "platform" in url_lower:
        return "platform"
    
    # Fallback to legacy category name
    legacy_lower = legacy_category.lower() if legacy_category else ""
    if "battery" in legacy_lower:
        return "battery_tightening"
    elif "cable" in legacy_lower:
        return "cable_tightening"
    elif "drill" in legacy_lower:
        return "electric_drilling"
    elif "platform" in legacy_lower:
        return "platform"
    
    return "unknown"


# =============================================================================
# PRODUCT FAMILY EXTRACTION
# =============================================================================

def extract_product_family(model_name: str, part_number: str = "") -> str:
    """
    Extract product family code from model name.
    
    Args:
        model_name: Product model name (e.g., "EPBC14-T4000-S6S4-T")
        part_number: Part number (fallback)
        
    Returns:
        Family code (e.g., "EPB", "EABS", "XPB")
    """
    if not model_name:
        return ""
    
    # Check for each family (order matters - longer patterns first)
    model_upper = model_name.upper()
    
    for family in PRODUCT_FAMILIES:
        if model_upper.startswith(family):
            # Special handling: EPBC -> EPB family
            if family == "EPBC":
                return "EPB"
            elif family in ("EABS", "EIBS"):
                return family  # Keep EABS/EIBS as distinct
            return family
    
    # Fallback: Extract first letters until digit
    match = re.match(r'^([A-Za-z]+)', model_name)
    if match:
        return match.group(1).upper()
    
    return ""


# =============================================================================
# TOOL TYPE DETECTION
# =============================================================================

def detect_tool_type(series_name: str, model_name: str, description: str = "") -> Optional[str]:
    """
    Detect tool type from series name, model name, or description.
    
    Args:
        series_name: Product series name
        model_name: Product model name
        description: Product description
        
    Returns:
        Tool type: pistol | angle_head | inline | screwdriver | drill | fixtured | None
    """
    # Combine all text sources
    text = f"{series_name} {model_name} {description}".lower()
    
    # Check each tool type pattern
    for tool_type, patterns in TOOL_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return tool_type
    
    return None


# =============================================================================
# WIRELESS DETECTION (Battery Tools Only)
# =============================================================================

def detect_wireless_info(
    model_name: str,
    description: str,
    html_content: str = "",
    legacy_wireless: str = "No"
) -> Dict:
    """
    Detect wireless capability for battery tools.
    
    Business Rules (priority order):
    1. If description contains "(standalone battery)" -> NOT wireless
    2. If model name has "C" before torque digits (EPBC14, EABS9C) -> WiFi capable
    3. If description mentions "Wireless", "Connect", "Smart Connected" -> WiFi capable
    4. Otherwise -> NOT wireless
    
    Args:
        model_name: Product model name
        description: Product description
        html_content: Full HTML content (for XPath-like searches)
        legacy_wireless: Legacy "Yes"/"No" wireless field
        
    Returns:
        WirelessInfo dict: {capable, detection_method, compatible_platforms}
    """
    wireless_info = {
        "capable": False,
        "detection_method": "not_applicable",
        "compatible_platforms": [],
        "compatible_platform_ids": []
    }
    
    # Combine text for searching
    all_text = f"{description} {html_content}".lower()
    model_upper = model_name.upper() if model_name else ""
    
    # ==========================================================================
    # RULE 1 (HIGHEST PRIORITY): Check for "C" in model name
    # If model has "C" in the family prefix, it IS WiFi capable - no exceptions
    # Examples: EPBC14, EABC10, EPBCHT, EPBACHT, EABSC9
    # ==========================================================================
    
    # WiFi model patterns - C after battery family prefix
    wifi_c_patterns = [
        r'^EPBC',       # EPBC14, EPBC8
        r'^EABC',       # EABC10
        r'^EABSC',      # EABSC9
        r'^EIBSC',      # EIBSC6
        r'^EPBCH',      # EPBCHT (High Torque with WiFi)
        r'^EPBACH',     # EPBACHT (Angle High Torque with WiFi)
        r'^EABCH',      # EABCHT
    ]
    
    for pattern in wifi_c_patterns:
        if re.match(pattern, model_upper):
            wireless_info["capable"] = True
            wireless_info["detection_method"] = "model_name_C"
            wireless_info["compatible_platforms"] = ["CVI3", "Connect"]
            logger.debug(f"  📶 {model_name}: WiFi capable (pattern: {pattern})")
            return wireless_info
    
    # ==========================================================================
    # RULE 2: Check for "wireless" keyword in description/specs
    # Example: "Compact, wireless, transducerized angle-head nutrunner"
    # But NOT if it says "standalone battery" - that means NO WiFi
    # ==========================================================================
    
    # First check if standalone battery - these are NEVER WiFi
    is_standalone = "standalone battery" in all_text or "standalone" in all_text
    
    if not is_standalone:
        wifi_keywords = ["wireless", "wifi", "wi-fi", "smart connected"]
        for keyword in wifi_keywords:
            if keyword in all_text:
                wireless_info["capable"] = True
                wireless_info["detection_method"] = "description_wireless"
                wireless_info["compatible_platforms"] = ["CVI3", "Connect"]
                logger.debug(f"  📶 {model_name}: WiFi capable (keyword: {keyword})")
                return wireless_info
    
    # ==========================================================================
    # RULE 3: Standalone battery = NO WiFi (explicit)
    # ==========================================================================
    
    if is_standalone:
        wireless_info["capable"] = False
        wireless_info["detection_method"] = "standalone_battery"
        logger.debug(f"  🔌 {model_name}: NOT wireless (standalone battery)")
        return wireless_info
    
    # ==========================================================================
    # RULE 4: Default - not wireless capable
    # Legacy field is NOT reliable, so we ignore it
    # ==========================================================================
    
    wireless_info["capable"] = False
    wireless_info["detection_method"] = "no_wifi_indicator"
    logger.debug(f"  🔌 {model_name}: NOT wireless (no indicators)")
    return wireless_info


# =============================================================================
# PLATFORM CONNECTION DETECTION (Cable Tools Only)
# =============================================================================

def detect_platform_connection(
    product_family: str,
    model_name: str,
    html_content: str = "",
    description: str = ""
) -> Dict:
    """
    Detect platform connection requirements for cable tools.
    
    Args:
        product_family: Product family code (EAD, EID, etc.)
        model_name: Product model name
        html_content: Full HTML content
        description: Product description
        
    Returns:
        PlatformConnection dict: {required, compatible_platforms, compatible_platform_ids}
    """
    platform_info = {
        "required": True,
        "compatible_platforms": [],
        "compatible_platform_ids": []
    }
    
    # Start with family-based platforms
    if product_family in CABLE_TOOL_PLATFORMS:
        platform_info["compatible_platforms"] = CABLE_TOOL_PLATFORMS[product_family].copy()
    
    # Enhance with content analysis
    all_text = f"{model_name} {description} {html_content}".upper()
    
    # Check for additional platform mentions
    platform_mentions = {
        "CVI3": ["CVI3", "CVIXS", "CVI-3"],
        "CVIR II": ["CVIR II", "CVIR-II", "CVIC II", "CVIC-II"],
        "ESP-C": ["ESP-C", "ESPC"],
        "ESP": ["ESP "],  # Space to avoid matching ESP-C
    }
    
    additional_platforms = []
    for platform, patterns in platform_mentions.items():
        for pattern in patterns:
            if pattern in all_text:
                if platform not in platform_info["compatible_platforms"]:
                    additional_platforms.append(platform)
                break
    
    platform_info["compatible_platforms"].extend(additional_platforms)
    
    # Log detection
    logger.debug(f"  🔌 {model_name}: Platforms: {platform_info['compatible_platforms']}")
    
    return platform_info


# =============================================================================
# MODULAR SYSTEM DETECTION (Electric Drilling Only)
# =============================================================================

def detect_modular_system(
    model_name: str,
    series_name: str,
    description: str = ""
) -> Dict:
    """
    Detect modular system info for electric drilling tools.
    
    Args:
        model_name: Product model name
        series_name: Product series name
        description: Product description
        
    Returns:
        ModularSystem dict: {is_base_tool, is_attachment, attachment_type, compatible_base_tools}
    """
    modular_info = {
        "is_base_tool": False,
        "is_attachment": False,
        "attachment_type": None,
        "compatible_base_tools": []
    }
    
    # Combine text for searching
    text = f"{model_name} {series_name} {description}".lower()
    
    # Check for base tools
    if "xpb-modular" in text or "xpb modular" in text:
        modular_info["is_base_tool"] = True
        modular_info["compatible_base_tools"] = ["XPB-Modular"]
        logger.debug(f"  🔧 {model_name}: Base tool (XPB-Modular)")
        return modular_info
    
    if "xpb-one" in text or "xpb one" in text:
        modular_info["is_base_tool"] = True
        modular_info["compatible_base_tools"] = ["XPB-One"]
        logger.debug(f"  🔧 {model_name}: Base tool (XPB-One)")
        return modular_info
    
    # Check for attachments
    if "tightening head" in text:
        modular_info["is_attachment"] = True
        modular_info["attachment_type"] = "tightening"
        modular_info["compatible_base_tools"] = MODULAR_BASE_TOOLS.copy()
        logger.debug(f"  🔩 {model_name}: Attachment (tightening head)")
        return modular_info
    
    if "drilling head" in text:
        modular_info["is_attachment"] = True
        modular_info["attachment_type"] = "drilling"
        modular_info["compatible_base_tools"] = MODULAR_BASE_TOOLS.copy()
        logger.debug(f"  🔩 {model_name}: Attachment (drilling head)")
        return modular_info
    
    # Generic XPB detection
    if model_name.upper().startswith("XPB"):
        modular_info["is_base_tool"] = True
        modular_info["compatible_base_tools"] = MODULAR_BASE_TOOLS.copy()
        logger.debug(f"  🔧 {model_name}: Base tool (XPB family)")
    
    return modular_info


# =============================================================================
# MAIN CATEGORIZATION FUNCTION
# =============================================================================

def categorize_product(
    model_name: str,
    part_number: str,
    series_name: str,
    category_url: str,
    legacy_category: str,
    description: str = "",
    html_content: str = "",
    legacy_wireless: str = "No"
) -> Dict:
    """
    Main function to categorize a product according to Schema v2.
    
    Args:
        model_name: Product model name
        part_number: Product part number
        series_name: Product series name
        category_url: Category page URL
        legacy_category: Legacy category string
        description: Product description
        html_content: Full HTML content for detailed analysis
        legacy_wireless: Legacy wireless field ("Yes"/"No")
        
    Returns:
        Dictionary with all Schema v2 categorization fields
    """
    logger.debug(f"\n📦 Categorizing: {model_name} ({part_number})")
    
    # Step 1: Detect tool category
    tool_category = detect_tool_category(category_url, legacy_category)
    logger.debug(f"  📁 Category: {tool_category}")
    
    # Step 2: Extract product family
    product_family = extract_product_family(model_name, part_number)
    logger.debug(f"  👨‍👩‍👧 Family: {product_family}")
    
    # Step 3: Detect tool type
    tool_type = detect_tool_type(series_name, model_name, description)
    logger.debug(f"  🔧 Type: {tool_type}")
    
    # Initialize category-specific fields
    wireless_info = None
    platform_connection = None
    modular_system = None
    
    # Step 4: Category-specific detection
    if tool_category == "battery_tightening":
        wireless_info = detect_wireless_info(
            model_name, description, html_content, legacy_wireless
        )
        logger.debug(f"  📶 Wireless: {wireless_info['capable']} ({wireless_info['detection_method']})")
        
    elif tool_category == "cable_tightening":
        platform_connection = detect_platform_connection(
            product_family, model_name, html_content, description
        )
        logger.debug(f"  🔌 Platforms: {platform_connection['compatible_platforms']}")
        
    elif tool_category == "electric_drilling":
        modular_system = detect_modular_system(model_name, series_name, description)
        if modular_system["is_base_tool"]:
            logger.debug(f"  🔧 Modular: Base tool")
        elif modular_system["is_attachment"]:
            logger.debug(f"  🔧 Modular: Attachment ({modular_system['attachment_type']})")
    
    # Build result
    result = {
        "tool_category": tool_category,
        "tool_type": tool_type,
        "product_family": product_family,
        "wireless": wireless_info,
        "platform_connection": platform_connection,
        "modular_system": modular_system,
        "schema_version": 2
    }
    
    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_standalone_battery(description: str, html_content: str = "") -> bool:
    """
    Check if product is a standalone battery tool (no WiFi).
    
    Args:
        description: Product description
        html_content: Full HTML content
        
    Returns:
        True if standalone battery tool
    """
    text = f"{description} {html_content}".lower()
    return "standalone battery" in text or "(standalone battery)" in text


def get_compatible_platforms_for_family(family: str) -> List[str]:
    """
    Get compatible platforms for a cable tool family.
    
    Args:
        family: Product family code
        
    Returns:
        List of compatible platform names
    """
    return CABLE_TOOL_PLATFORMS.get(family, [])
