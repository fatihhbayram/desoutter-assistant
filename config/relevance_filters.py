"""
Relevance filtering configuration for RAG system.
Defines negative keyword rules to exclude irrelevant documents.

PRODUCTION-SAFE:
- Can be disabled via ENABLE_RELEVANCE_FILTERING flag
- Additive only - does not modify existing behavior when disabled
- Config-driven - no hardcoded values in logic

EXPANDED: Now covers 15 fault categories instead of 5
"""

# Enable/disable relevance filtering (set to False to revert to original behavior)
ENABLE_RELEVANCE_FILTERING = True

# Negative keyword rules per fault category
# Each rule defines:
# - trigger_keywords: Keywords that identify this fault category
# - exclude_if_contains: Documents containing these are excluded
# - require_at_least_one: Documents must contain at least one of these
RELEVANCE_FILTER_RULES = {
    # Network & Connectivity
    "wifi_network": {
        "trigger_keywords": ["wifi", "wireless", "network", "connect unit", "access point", "communication"],
        "exclude_if_contains": [
            "transducer wire", "transducer rework", "torque calibration", "torque sensor",
            "motor replacement", "gearbox", "spindle", "bearing wear"
        ],
        "require_at_least_one": [
            "wifi", "wireless", "network", "connect", "access point", "communication", "signal"
        ]
    },
    
    # Mechanical Issues
    "motor_mechanical": {
        "trigger_keywords": ["motor", "spindle", "rotation", "gearbox", "bearing", "mechanical"],
        "exclude_if_contains": [
            "wifi", "wireless", "network configuration", "software update", "firmware",
            "connect unit", "access point", "display", "touchscreen"
        ],
        "require_at_least_one": [
            "motor", "spindle", "mechanical", "bearing", "gearbox", "rotation"
        ]
    },
    
    # Measurement & Calibration
    "torque_calibration": {
        "trigger_keywords": ["torque", "calibration", "transducer", "accuracy", "measurement"],
        "exclude_if_contains": [
            "wifi", "network", "motor replacement", "battery charging", "connect unit",
            "display", "touchscreen"
        ],
        "require_at_least_one": [
            "torque", "calibration", "transducer", "accuracy", "measurement", "sensor"
        ]
    },
    
    # Power & Battery
    "battery_power": {
        "trigger_keywords": ["battery", "charging", "power", "voltage", "charger"],
        "exclude_if_contains": [
            "wifi", "network", "motor", "torque calibration", "transducer", "gearbox",
            "display", "touchscreen"
        ],
        "require_at_least_one": [
            "battery", "power", "charging", "voltage", "charge", "charger"
        ]
    },
    
    # Software & Firmware
    "software_firmware": {
        "trigger_keywords": ["software", "firmware", "update", "version", "installation"],
        "exclude_if_contains": [
            "motor replacement", "bearing wear", "gearbox", "transducer wire"
        ],
        "require_at_least_one": [
            "software", "firmware", "update", "version", "installation", "upgrade"
        ]
    },
    
    # Display Issues
    "display_screen": {
        "trigger_keywords": ["display", "screen", "lcd", "no display", "blank screen", "flickering"],
        "exclude_if_contains": [
            "motor", "gearbox", "torque calibration", "transducer wire", "battery charging"
        ],
        "require_at_least_one": [
            "display", "screen", "lcd", "visual", "monitor"
        ]
    },
    
    # Touchscreen Issues
    "touchscreen": {
        "trigger_keywords": ["touchscreen", "touch", "screen not responding", "touch panel"],
        "exclude_if_contains": [
            "motor", "gearbox", "torque calibration", "transducer wire", "wifi"
        ],
        "require_at_least_one": [
            "touchscreen", "touch", "screen", "panel", "responsive"
        ]
    },
    
    # Pset & Configuration
    "pset_configuration": {
        "trigger_keywords": ["pset", "parameter", "configuration", "settings", "program"],
        "exclude_if_contains": [
            "motor replacement", "bearing wear", "gearbox", "transducer wire"
        ],
        "require_at_least_one": [
            "pset", "parameter", "configuration", "settings", "program", "setup"
        ]
    },
    
    # Sensor Issues
    "sensor": {
        "trigger_keywords": ["sensor", "detection", "proximity", "position sensor"],
        "exclude_if_contains": [
            "wifi", "network", "motor replacement", "display"
        ],
        "require_at_least_one": [
            "sensor", "detection", "proximity", "position", "sensing"
        ]
    },
    
    # Error Codes
    "error_codes": {
        "trigger_keywords": [
            "error", "fault code", "alarm", "warning", "alert",
            # Common Desoutter error codes
            "e01", "e02", "e03", "e04", "e05", "e06", "e07", "e08", "e09",
            "e10", "e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18", "e018",
            "e19", "e20", "e21", "e22", "e23", "e24", "e25",
            # Transducer-related
            "transducer", "transducer fault", "transducer error"
        ],
        "exclude_if_contains": [],  # Don't exclude anything - error codes can relate to any system
        "require_at_least_one": [
            "error", "fault", "alarm", "warning", "code", "alert", "transducer",
            "symptom", "issue", "problem", "solution"
        ]
    },
    
    # Sound & Noise
    "sound_noise": {
        "trigger_keywords": ["sound", "noise", "beep", "alarm", "buzzer", "silent"],
        "exclude_if_contains": [
            "motor replacement", "gearbox", "bearing wear", "wifi", "display"
        ],
        "require_at_least_one": [
            "sound", "noise", "beep", "alarm", "buzzer", "audio"
        ]
    },
    
    # Communication & Protocol
    "communication_protocol": {
        "trigger_keywords": ["communication", "protocol", "ethernet", "fieldbus", "modbus"],
        "exclude_if_contains": [
            "motor replacement", "bearing wear", "gearbox", "transducer wire", "display"
        ],
        "require_at_least_one": [
            "communication", "protocol", "ethernet", "fieldbus", "modbus", "data"
        ]
    },
    
    # LED & Indicators
    "led_indicators": {
        "trigger_keywords": ["led", "light", "indicator", "blinking", "flashing"],
        "exclude_if_contains": [
            "motor", "gearbox", "torque calibration", "transducer wire"
        ],
        "require_at_least_one": [
            "led", "light", "indicator", "blinking", "flashing", "lamp"
        ]
    },
    
    # Button & Controls
    "button_controls": {
        "trigger_keywords": ["button", "switch", "trigger", "control", "start button"],
        "exclude_if_contains": [
            "motor", "gearbox", "torque calibration", "wifi", "network"
        ],
        "require_at_least_one": [
            "button", "switch", "trigger", "control", "press"
        ]
    },
    
    # Cable & Connector
    "cable_connector": {
        "trigger_keywords": ["cable", "connector", "plug", "wire", "wiring", "cable damage", "loose cable"],
        "exclude_if_contains": [
            "motor replacement", "gearbox", "display", "touchscreen", "wifi", "network"
        ],
        "require_at_least_one": [
            "cable", "connector", "plug", "wire", "wiring"
        ]
    }
}

# Minimum similarity threshold for filtered results
# Documents below this threshold are excluded even if they pass keyword filters
MIN_SIMILARITY_AFTER_FILTER = 0.25

# Maximum number of documents to exclude (safety limit)
# If more than this many would be excluded, filtering is skipped
MAX_DOCUMENTS_TO_EXCLUDE = 10
