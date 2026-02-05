"""
Stage 4: Knowledge Graph Validation

Validates extracted information against hard-coded compatibility matrix
and parameter ranges.

Responsibilities:
- Validate tool-controller compatibility
- Check parameter ranges (min/max values)
- Detect incompatible configurations
- Add warnings/blocks for invalid information
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

from .stage1_intent_classifier import IntentType, IntentResult
from .stage3_info_extraction import ExtractionResult

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation result status"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    UNKNOWN = "unknown"


@dataclass
class ValidationIssue:
    """A single validation issue"""
    field: str
    status: ValidationStatus
    message: str
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of knowledge graph validation"""
    status: ValidationStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_data: Optional[ExtractionResult] = None
    compatibility_verified: bool = False
    parameter_ranges_verified: bool = False
    confidence_adjustment: float = 0.0


class KGValidator:
    """
    Stage 4: Knowledge Graph Validator
    
    Validates extracted information against hard-coded compatibility
    matrix and parameter ranges.
    """
    
    # Hard-coded compatibility matrix
    # Format: {product_prefix: {controllers, firmware, parameter_ranges, accessories}}
    COMPATIBILITY_MATRIX = {
        # EABC Series (Advanced Battery Cordless)
        "EABC": {
            "controllers": {
                "CVI3": {"min_version": "2.5", "max_version": None, "recommended": True},
                "CVIR": {"min_version": "3.0", "max_version": None, "recommended": True},
                "CVIC-II": {"min_version": "1.8", "max_version": "2.5", "recommended": False},
                "CONNECT-W": {"min_version": "1.0", "max_version": None, "recommended": True},
                "CONNECT-X": {"min_version": "1.0", "max_version": None, "recommended": True},
            },
            "firmware": {
                "tool_min": "1.8",
                "controller_min": "2.5",
            },
            "parameter_ranges": {
                "torque": {"min": 5, "max": 85, "unit": "Nm"},
                "angle": {"min": 0, "max": 999, "unit": "degrees"},
                "speed": {"min": 50, "max": 1500, "unit": "RPM"},
            },
            "accessories": {
                "docks": ["Dock-Alpha", "Dock-Beta", "CDED3-4"],
                "batteries": ["BAT-2000", "BAT-2500", "BAT-3000"],
                "cables": ["USB-C", "WiFi"],
            },
            "capabilities": ["WiFi", "Bluetooth", "Display", "DataLogging"],
        },
        
        # EABS Series (Standard Battery)
        "EABS": {
            "controllers": {
                "CVI3": {"min_version": "2.0", "max_version": None, "recommended": True},
                "CVIC-II": {"min_version": "1.5", "max_version": None, "recommended": True},
            },
            "firmware": {
                "tool_min": "1.5",
                "controller_min": "2.0",
            },
            "parameter_ranges": {
                "torque": {"min": 2, "max": 45, "unit": "Nm"},
                "angle": {"min": 0, "max": 720, "unit": "degrees"},
                "speed": {"min": 100, "max": 1200, "unit": "RPM"},
            },
            "accessories": {
                "docks": ["Dock-Basic", "CDED3-2"],
                "batteries": ["BAT-1500", "BAT-2000"],
                "cables": ["USB-A"],
            },
            "capabilities": ["Display", "DataLogging"],
        },
        
        # EFD Series (Electric Fixed Drive / Corded Screwdriver)
        "EFD": {
            "controllers": {
                "CVI3": {"min_version": "2.0", "max_version": None, "recommended": True},
                "CVIL-II": {"min_version": "1.0", "max_version": None, "recommended": True},
                "CVIR-II": {"min_version": "2.0", "max_version": None, "recommended": False},
            },
            "firmware": {
                "tool_min": "1.2",
                "controller_min": "2.0",
            },
            "parameter_ranges": {
                "torque": {"min": 0.5, "max": 12, "unit": "Nm"},
                "angle": {"min": 0, "max": 999, "unit": "degrees"},
                "speed": {"min": 200, "max": 2000, "unit": "RPM"},
            },
            "accessories": {
                "cables": ["CVI3-Cable", "Standard-Cable"],
                "bits": ["Standard", "Torx", "Phillips", "Hex"],
            },
            "capabilities": ["Corded", "HighPrecision"],
        },
        
        # EPBAHT Series (Pneumatic Battery Angle Head Tool)
        "EPBAHT": {
            "controllers": {
                "CVI3": {"min_version": "2.5", "max_version": None, "recommended": True},
                "CVIR": {"min_version": "3.0", "max_version": None, "recommended": True},
            },
            "firmware": {
                "tool_min": "2.0",
                "controller_min": "2.5",
            },
            "parameter_ranges": {
                "torque": {"min": 10, "max": 120, "unit": "Nm"},
                "angle": {"min": 0, "max": 999, "unit": "degrees"},
                "speed": {"min": 50, "max": 800, "unit": "RPM"},
            },
            "accessories": {
                "docks": ["Dock-HeavyDuty"],
                "batteries": ["BAT-3000", "BAT-4000"],
            },
            "capabilities": ["AngleHead", "HighTorque", "DataLogging"],
        },
        
        # ERSF Series (Electric Rotary Spindle Fixed)
        "ERSF": {
            "controllers": {
                "CVIR-II": {"min_version": "2.0", "max_version": None, "recommended": True},
                "CVI3": {"min_version": "2.5", "max_version": None, "recommended": False},
            },
            "firmware": {
                "tool_min": "1.5",
                "controller_min": "2.0",
            },
            "parameter_ranges": {
                "torque": {"min": 1, "max": 25, "unit": "Nm"},
                "angle": {"min": 0, "max": 999, "unit": "degrees"},
                "speed": {"min": 100, "max": 1500, "unit": "RPM"},
            },
            "accessories": {
                "cables": ["CVIR-Cable"],
                "adapters": ["ERS-Adapter"],
            },
            "capabilities": ["Fixed", "HighSpeed"],
        },
        
        # EPBC Series (Battery Cordless with WiFi)
        "EPBC": {
            "controllers": {
                "CVI3": {"min_version": "2.5", "max_version": None, "recommended": True},
                "CONNECT-W": {"min_version": "1.0", "max_version": None, "recommended": True},
                "CONNECT-X": {"min_version": "1.0", "max_version": None, "recommended": True},
            },
            "firmware": {
                "tool_min": "2.0",
                "controller_min": "2.5",
            },
            "parameter_ranges": {
                "torque": {"min": 3, "max": 60, "unit": "Nm"},
                "angle": {"min": 0, "max": 999, "unit": "degrees"},
                "speed": {"min": 100, "max": 1400, "unit": "RPM"},
            },
            "accessories": {
                "docks": ["Dock-WiFi", "CDED3-4"],
                "batteries": ["BAT-2000", "BAT-2500"],
            },
            "capabilities": ["WiFi", "Bluetooth", "DataLogging"],
        },
    }
    
    # Controller version compatibility
    CONTROLLER_INFO = {
        "CVI3": {
            "current_version": "3.2",
            "min_supported": "2.0",
            "features": ["WiFi", "Ethernet", "USB", "Bluetooth", "Pset", "Strategy"],
        },
        "CVIR": {
            "current_version": "3.5",
            "min_supported": "2.5",
            "features": ["Ethernet", "USB", "MultiChannel", "Pset", "Strategy"],
        },
        "CVIR-II": {
            "current_version": "3.0",
            "min_supported": "1.5",
            "features": ["Ethernet", "USB", "MultiTool", "Pset"],
        },
        "CVIC-II": {
            "current_version": "2.8",
            "min_supported": "1.0",
            "features": ["USB", "Pset", "Basic"],
        },
        "CVIL-II": {
            "current_version": "2.5",
            "min_supported": "1.0",
            "features": ["USB", "Pset", "TorqueControl"],
        },
        "CONNECT-W": {
            "current_version": "1.5",
            "min_supported": "1.0",
            "features": ["WiFi-AP", "Cloud", "Wireless"],
        },
        "CONNECT-X": {
            "current_version": "1.5",
            "min_supported": "1.0",
            "features": ["WiFi-External", "Cloud", "Wireless"],
        },
    }
    
    def __init__(self):
        """Initialize KG Validator"""
        pass
    
    def validate(
        self,
        extraction_result: ExtractionResult,
        intent_result: IntentResult
    ) -> ValidationResult:
        """
        Validate extraction result against knowledge graph.
        
        Args:
            extraction_result: Result from Stage 3
            intent_result: Result from Stage 1
            
        Returns:
            ValidationResult with issues and status
        """
        issues = []
        
        product = extraction_result.product_model
        intent = intent_result.primary_intent
        
        logger.info(f"Validating extraction for: {product}, intent: {intent.value}")
        
        # Get product family
        product_family = self._get_product_family(product)
        
        # Validate based on intent
        if intent == IntentType.COMPATIBILITY:
            comp_issues = self._validate_compatibility(
                extraction_result, product_family
            )
            issues.extend(comp_issues)
        
        if intent == IntentType.CONFIGURATION:
            config_issues = self._validate_configuration(
                extraction_result, product_family, intent_result
            )
            issues.extend(config_issues)
        
        if intent in [IntentType.TROUBLESHOOT, IntentType.ERROR_CODE]:
            trouble_issues = self._validate_troubleshoot(
                extraction_result, product_family
            )
            issues.extend(trouble_issues)
        
        # Determine overall status
        overall_status = self._determine_status(issues)
        
        # Calculate confidence adjustment
        confidence_adj = self._calculate_confidence_adjustment(issues)
        
        result = ValidationResult(
            status=overall_status,
            issues=issues,
            validated_data=extraction_result,
            compatibility_verified=any(
                i.field == "compatibility" for i in issues
            ),
            parameter_ranges_verified=any(
                i.field == "parameter_range" for i in issues
            ),
            confidence_adjustment=confidence_adj,
        )
        
        logger.info(f"Validation complete: {overall_status.value}, {len(issues)} issues")
        
        return result
    
    def _get_product_family(self, product_model: Optional[str]) -> Optional[str]:
        """Extract product family from model number"""
        if not product_model:
            return None
        
        # Try to match known prefixes
        for prefix in self.COMPATIBILITY_MATRIX.keys():
            if product_model.upper().startswith(prefix):
                return prefix
        
        return None
    
    def _validate_compatibility(
        self,
        extraction: ExtractionResult,
        product_family: Optional[str]
    ) -> List[ValidationIssue]:
        """Validate compatibility information"""
        issues = []
        
        if not product_family:
            issues.append(ValidationIssue(
                field="product",
                status=ValidationStatus.WARN,
                message="Ürün ailesi tanımlanamadı, uyumluluk doğrulanamıyor",
                suggestion="Ürün modelini tam olarak belirtin"
            ))
            return issues
        
        matrix = self.COMPATIBILITY_MATRIX.get(product_family)
        if not matrix:
            issues.append(ValidationIssue(
                field="compatibility",
                status=ValidationStatus.UNKNOWN,
                message=f"{product_family} için uyumluluk verisi bulunamadı",
            ))
            return issues
        
        # Validate extracted compatibility against matrix
        if extraction.compatibility:
            for ctrl in extraction.compatibility.compatible_controllers:
                ctrl_model = ctrl.get("model", "")
                ctrl_version = ctrl.get("min_version", "")
                
                if ctrl_model in matrix["controllers"]:
                    expected = matrix["controllers"][ctrl_model]
                    
                    # Check version
                    if expected["min_version"]:
                        if self._compare_versions(ctrl_version, expected["min_version"]) < 0:
                            issues.append(ValidationIssue(
                                field="compatibility",
                                status=ValidationStatus.BLOCK,
                                message=f"{ctrl_model} minimum versiyon {expected['min_version']} gerekli",
                                suggestion=f"{ctrl_model} v{expected['min_version']}+ kullanın",
                                details={"controller": ctrl_model, "required_version": expected["min_version"]}
                            ))
                        else:
                            issues.append(ValidationIssue(
                                field="compatibility",
                                status=ValidationStatus.ALLOW,
                                message=f"{ctrl_model} v{ctrl_version} uyumlu",
                            ))
                else:
                    # Controller not in our matrix - warn
                    issues.append(ValidationIssue(
                        field="compatibility",
                        status=ValidationStatus.WARN,
                        message=f"{ctrl_model} uyumluluğu doğrulanamadı",
                        suggestion="Resmi dokümantasyonu kontrol edin"
                    ))
        
        # Add known compatible controllers from matrix
        known_controllers = list(matrix["controllers"].keys())
        issues.append(ValidationIssue(
            field="compatibility",
            status=ValidationStatus.ALLOW,
            message=f"Bilinen uyumlu controller'lar: {', '.join(known_controllers)}",
            details={"known_controllers": known_controllers}
        ))
        
        return issues
    
    def _validate_configuration(
        self,
        extraction: ExtractionResult,
        product_family: Optional[str],
        intent_result: IntentResult
    ) -> List[ValidationIssue]:
        """Validate configuration parameters"""
        issues = []
        
        if not product_family:
            return issues
        
        matrix = self.COMPATIBILITY_MATRIX.get(product_family)
        if not matrix:
            return issues
        
        # Check parameter ranges
        param_type = intent_result.entities.parameter_type
        target_value = intent_result.entities.target_value
        
        if param_type and target_value:
            ranges = matrix.get("parameter_ranges", {}).get(param_type)
            
            if ranges:
                # Parse numeric value
                numeric_value = self._parse_numeric(target_value)
                
                if numeric_value is not None:
                    min_val = ranges.get("min", 0)
                    max_val = ranges.get("max", float("inf"))
                    unit = ranges.get("unit", "")
                    
                    if numeric_value < min_val:
                        issues.append(ValidationIssue(
                            field="parameter_range",
                            status=ValidationStatus.BLOCK,
                            message=f"{param_type} değeri ({numeric_value}) minimum değerin ({min_val} {unit}) altında",
                            suggestion=f"Minimum {min_val} {unit} kullanın",
                            details={"parameter": param_type, "value": numeric_value, "min": min_val, "max": max_val}
                        ))
                    elif numeric_value > max_val:
                        issues.append(ValidationIssue(
                            field="parameter_range",
                            status=ValidationStatus.BLOCK,
                            message=f"{param_type} değeri ({numeric_value}) maksimum değerin ({max_val} {unit}) üstünde",
                            suggestion=f"Maksimum {max_val} {unit} kullanın",
                            details={"parameter": param_type, "value": numeric_value, "min": min_val, "max": max_val}
                        ))
                    elif numeric_value > max_val * 0.9:
                        issues.append(ValidationIssue(
                            field="parameter_range",
                            status=ValidationStatus.WARN,
                            message=f"{param_type} değeri ({numeric_value} {unit}) maksimuma yakın",
                            suggestion="Daha düşük bir değer düşünün",
                            details={"parameter": param_type, "value": numeric_value, "max": max_val}
                        ))
                    else:
                        issues.append(ValidationIssue(
                            field="parameter_range",
                            status=ValidationStatus.ALLOW,
                            message=f"{param_type} değeri ({numeric_value} {unit}) izin verilen aralıkta ({min_val}-{max_val} {unit})",
                        ))
        
        # Validate controller requirement from prerequisites
        if extraction.prerequisites and extraction.prerequisites.controller:
            ctrl = extraction.prerequisites.controller
            ctrl_model = ctrl.get("model", "")
            
            if ctrl_model and ctrl_model in matrix.get("controllers", {}):
                expected = matrix["controllers"][ctrl_model]
                if expected.get("recommended"):
                    issues.append(ValidationIssue(
                        field="controller",
                        status=ValidationStatus.ALLOW,
                        message=f"{ctrl_model} bu ürün için önerilen controller",
                    ))
        
        return issues
    
    def _validate_troubleshoot(
        self,
        extraction: ExtractionResult,
        product_family: Optional[str]
    ) -> List[ValidationIssue]:
        """Validate troubleshooting information"""
        issues = []
        
        if extraction.troubleshoot:
            # Check if ESDE reference exists
            if extraction.troubleshoot.esde_reference:
                issues.append(ValidationIssue(
                    field="esde",
                    status=ValidationStatus.WARN,
                    message=f"Bilinen üretim hatası tespit edildi: {extraction.troubleshoot.esde_reference}",
                    suggestion="Servis müdahalesi gerekebilir",
                    details={"esde_code": extraction.troubleshoot.esde_reference}
                ))
            
            # Check affected models
            if extraction.troubleshoot.affected_models:
                issues.append(ValidationIssue(
                    field="affected_models",
                    status=ValidationStatus.WARN,
                    message=f"Etkilenen modeller: {', '.join(extraction.troubleshoot.affected_models)}",
                ))
        
        return issues
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare version strings.
        Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        if not v1:
            return -1
        if not v2:
            return 1
        
        try:
            parts1 = [int(x) for x in v1.split(".")]
            parts2 = [int(x) for x in v2.split(".")]
            
            # Pad shorter version
            while len(parts1) < len(parts2):
                parts1.append(0)
            while len(parts2) < len(parts1):
                parts2.append(0)
            
            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            return 0
        except ValueError:
            return 0
    
    def _parse_numeric(self, value: str) -> Optional[float]:
        """Parse numeric value from string"""
        if not value:
            return None
        
        match = re.search(r'(\d+(?:\.\d+)?)', value)
        if match:
            return float(match.group(1))
        return None
    
    def _determine_status(self, issues: List[ValidationIssue]) -> ValidationStatus:
        """Determine overall validation status"""
        if any(i.status == ValidationStatus.BLOCK for i in issues):
            return ValidationStatus.BLOCK
        if any(i.status == ValidationStatus.WARN for i in issues):
            return ValidationStatus.WARN
        if any(i.status == ValidationStatus.UNKNOWN for i in issues):
            return ValidationStatus.UNKNOWN
        return ValidationStatus.ALLOW
    
    def _calculate_confidence_adjustment(self, issues: List[ValidationIssue]) -> float:
        """Calculate confidence adjustment based on issues"""
        adjustment = 0.0
        
        for issue in issues:
            if issue.status == ValidationStatus.ALLOW:
                adjustment += 0.05  # Boost for verified info
            elif issue.status == ValidationStatus.WARN:
                adjustment -= 0.1  # Slight penalty for warnings
            elif issue.status == ValidationStatus.BLOCK:
                adjustment -= 0.3  # Strong penalty for blocks
        
        return max(-0.5, min(0.2, adjustment))  # Clamp adjustment
    
    def get_product_info(self, product_model: str) -> Optional[Dict[str, Any]]:
        """Get known information about a product"""
        family = self._get_product_family(product_model)
        if family and family in self.COMPATIBILITY_MATRIX:
            return self.COMPATIBILITY_MATRIX[family]
        return None
    
    def get_controller_info(self, controller: str) -> Optional[Dict[str, Any]]:
        """Get known information about a controller"""
        return self.CONTROLLER_INFO.get(controller)
    
    def check_tool_controller_compatibility(
        self,
        product_model: str,
        controller: str,
        controller_version: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Quick check for tool-controller compatibility.
        
        Returns:
            (is_compatible, message)
        """
        family = self._get_product_family(product_model)
        
        if not family:
            return (False, f"Ürün ailesi tanımlanamadı: {product_model}")
        
        matrix = self.COMPATIBILITY_MATRIX.get(family)
        if not matrix:
            return (False, f"Uyumluluk verisi yok: {family}")
        
        controllers = matrix.get("controllers", {})
        
        if controller not in controllers:
            compatible_list = list(controllers.keys())
            return (False, f"{product_model} {controller} ile uyumlu değil. Uyumlu: {', '.join(compatible_list)}")
        
        ctrl_info = controllers[controller]
        
        if controller_version and ctrl_info.get("min_version"):
            if self._compare_versions(controller_version, ctrl_info["min_version"]) < 0:
                return (False, f"{controller} minimum v{ctrl_info['min_version']} gerekli, mevcut: v{controller_version}")
        
        recommended = "✓ Önerilen" if ctrl_info.get("recommended") else ""
        return (True, f"{product_model} + {controller}: Uyumlu {recommended}")
