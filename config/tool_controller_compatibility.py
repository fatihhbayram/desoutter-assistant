"""
Tool-Controller Compatibility Mapping

Based on controller user manuals and technical documentation.
This mapping helps determine which tools are compatible with which controllers.

IMPORTANT: This mapping is based on actual controller manuals and user input.
Do not modify without verifying against official documentation.
"""

# Tool-Controller Compatibility Database
# Format: {controller_series: {compatible_tool_patterns, notes, source}}

TOOL_CONTROLLER_COMPATIBILITY = {
    "CVIL II": {
        "compatible_tool_patterns": [
            # Angle head
            r"^ERAL",
            # In-line
            r"^ERDL",
            # Pistol grip
            r"^ERPL", r"^ERPS", r"^ERPLT",
            # Fixed tools
            r"^EME", r"^EMEL", r"^EMEO",
            # Pulse range
            r"^ELRT",
        ],
        "incompatible_notes": "Tools with second torque transducer not supported",
        "description": "Complete range of Torque Control tools",
        "source": "CVIL II_user manual_English_6159933780_EN.pdf - Section 2.6"
    },
    
    "CVI3": {
        "compatible_tool_patterns": [
            # Battery tools (non-WiFi)
            r"^EPB", r"^EPBA", r"^EPBHT", r"^EPBAHT",
            # Battery tools with WiFi
            r"^EPBC", r"^EPBCHT",
            # Advanced battery tools
            r"^EAB", r"^EABA", r"^EABS",
            # Advanced battery with WiFi
            r"^EABC",
            # Battery pulse tools
            r"^BLRT",
            # XPB series
            r"^XPB",
            # Corded screwdrivers (EFD is corded, works with CVI3)
            r"^EFD", r"^EFDE", r"^EFDA", r"^EFDO",
            r"^EFM",
            # ERS series (with adapter/converter)
            r"^ERS", r"^ERSA",
        ],
        "description": "Battery-powered and corded tools",
        "notes": "ERS series requires CVI3 adapter/converter",
        "source": "CVI3 - CVILogix Library - Manual - V2.2.1.x.pdf"
    },
    
    "CONNECT-W": {
        "compatible_tool_patterns": [
            r"^EPBC", r"^EPBCHT",  # WiFi battery tools only
            r"^EABC",
        ],
        "description": "Wireless communication tools only (built-in AP)",
        "notes": "CONNECT-W has built-in WiFi access point",
        "source": "ConnectManuel.pdf"
    },
    
    "CONNECT-X": {
        "compatible_tool_patterns": [
            r"^EPBC", r"^EPBCHT",  # WiFi battery tools only
            r"^EABC",
        ],
        "description": "Wireless communication tools only (requires external AP)",
        "notes": "CONNECT-X requires external WiFi access point",
        "source": "ConnectManuel.pdf"
    },
    
    "CVIR II": {
        "compatible_tool_patterns": [
            # Handheld tools
            r"^ECP",    # ECP3L, ECP5L, ECP10L, ECP20L, ECP3LT, ECP5LT, ECP10LT, ECP20LT, ECP5
            r"^ECL",    # ECL1, ECL3, ECL5, ECL8, ECL11
            r"^ECLA",   # ECLA1, ECLA3, ECLA5, ECLA8, ECLA11
            r"^ECD",    # ECD5
            r"^ECA",    # ECA15
            r"^ECS",    # ECS06, ECS2, ECS4, ECS7, ECS10, ECS16 (+ M20 variants)
            r"^ECSA",   # ECSA2, ECSA7, ECSA10
            r"^ERS",    # ERS2, ERS6, ERS12 (+ M20 variants)
            r"^ERSA",   # ERSA2, ERSA6, ERSA12
            r"^MC",     # MC35-10
            # Fixed tools
            r"^ECSF",   # ECSF06, ECSF2, ECSF4, ECSF7, ECSF10, ECSF16
            r"^ECF",    # ECF3L, ECF5L, ECF10L, ECF20L
            r"^ERSF",   # ERSF2, ERSF7, ERSF10
            # ERPHT mode
            r"^ERP",    # ERP250, ERP500, ERP750, ERP1000, ERP170
        ],
        "description": "Electric torque control and clutch tools",
        "notes": "Supports both Normal mode and ERPHT mode",
        "source": "CVIR II_user manual_English_6159933910_EN.pdf"
    },
    
    "CVIC II": {
        "compatible_tool_patterns": [
            # Normal mode - Handheld tools
            r"^ECP",    # ECP3L, ECP5L, ECP10L, ECP20L, ECP3LT, ECP5LT, ECP10LT, ECP20LT, ECP5
            r"^ECL",    # ECL1, ECL3, ECL5, ECL8, ECL11, ECLA1, ECLA3, ECLA5, ECLA8, ECLA11
            r"^ECLA",
            r"^ECD",    # ECD5, ECD20, ECD30, ECD50, ECD70, ECD120
            r"^ECA",    # ECA15, ECA20, ECA30, ECA40, ECA60, ECA70, ECA90, ECA115, ECA125, ECA150, ECA200
            r"^ECS",    # ECS06, ECS2, ECS4, ECS7, ECS10, ECS16 (+ M20 variants)
            r"^ECSA",   # ECSA2, ECSA7, ECSA10
            # Normal mode - Fixed tools
            r"^MC",     # MC35-10, MC35-20, MC38-10, MC38-20, MC51-10, MC51-20, MC60-10, MC60-20, MC60-30, MC80-10, MC80-20, MC80-30, MC80-40, MC106-10, MC106-20
            r"^MCL",    # MCL38-20, MCL51-20, MCL60-20, MCL60-30, MCL80-40
            r"^ECSF",   # ECSF06, ECSF2, ECSF4, ECSF7, ECSF10, ECSF16
            r"^ECF",    # ECF3L, ECF5L, ECF10L, ECF20L, ECF20S, ECF30S
            # ECPHT mode - High Torque
            r"^ECPHT",  # ECPHT, ECP190, ECP550, ECP950, ECP1500, ECP2100, ECP3000, ECP4000, ECP100R, ECP190R, ECP550R, ECP950R
        ],
        "description": "Mechanical and electric clutch tools",
        "notes": "Supports Normal mode (L2/H2) and ECPHT mode (L4/H4)",
        "source": "CVIC II_user manual_English_6159932190_EN.pdf"
    },
    
    "CVIXS": {
        "compatible_tool_patterns": [
            # Handheld tools
            r"^ERXS",   # ERXS20, ERXS50, ERXS80
        ],
        "description": "Simplified torque control - ERXS series",
        "source": "CVIXS controller_V 5.5.X_User manual_EN_6159939280_EN.pdf"
    },
    
    "ESP": {
        "compatible_tool_patterns": [
            # SLC series (corded screwdrivers)
            r"^SLC", r"^SLBN",
        ],
        "description": "Low voltage electric screwdrivers - SLC series",
        "source": "ESP-C_User_Manual_6159934960.pdf"
    },
    
    "AXON": {
        "compatible_tool_patterns": [
            # Handheld tools
            r"^EAD",    # Angle head range
            r"^ERSA",   # Angle head (with ERS module adapter)
            r"^EID", r"^EIDS",  # Inline range
            r"^ERS",    # Inline range (with ERS module adapter)
            r"^EPD",    # Pistol range (including EPD-LRT)
            # Fixtured tools (spindle range)
            r"^EFDE", r"^EFDS", r"^EFDA", r"^EFDO",
            r"^ERSF",   # Spindle (with ERS module adapter)
            # WiFi tools
            r"^EPBC", r"^EPBCHT",
            r"^EABC",
        ],
        "description": "Most Desoutter electric tools + WiFi tools",
        "notes": "Tools with * require ERS module adapter. EFD-TA tools available soon.",
        "source": "AXON manual - List of compatible tightening tools"
    },
}


def get_compatible_controllers(tool_model: str) -> list:
    """
    Get list of compatible controllers for a given tool model.
    
    Args:
        tool_model: Tool model name (e.g., "EPBC8-1800-4Q")
        
    Returns:
        List of compatible controller names
    """
    import re
    
    compatible = []
    
    for controller, info in TOOL_CONTROLLER_COMPATIBILITY.items():
        patterns = info.get("compatible_tool_patterns", [])
        for pattern in patterns:
            if re.match(pattern, tool_model, re.IGNORECASE):
                compatible.append(controller)
                break
    
    return compatible


def get_compatible_tools(controller_name: str) -> dict:
    """
    Get compatibility info for a given controller.
    
    Args:
        controller_name: Controller name (e.g., "CVIL II")
        
    Returns:
        Dictionary with compatibility info or None if not found
    """
    return TOOL_CONTROLLER_COMPATIBILITY.get(controller_name)


def check_compatibility(tool_model: str, controller_name: str) -> bool:
    """
    Check if a tool is compatible with a controller.
    
    Args:
        tool_model: Tool model name
        controller_name: Controller name
        
    Returns:
        True if compatible, False otherwise
    """
    compatible_controllers = get_compatible_controllers(tool_model)
    return controller_name in compatible_controllers
