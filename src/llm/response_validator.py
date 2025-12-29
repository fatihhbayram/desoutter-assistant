"""
Response Validation Module
Post-processes LLM responses to detect potential hallucinations and quality issues.

Validation Checks:
1. Uncertainty phrase detection ("might", "probably", "could be")
2. Numerical value verification (numbers must exist in context)
3. Response length check (too short = insufficient)
4. Product mismatch detection (mentions wrong product)
5. Forbidden content detection (suggestions outside capabilities)
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ValidationIssue:
    """Single validation issue found in response"""
    type: str  # "uncertainty", "number_mismatch", "short_response", etc.
    description: str  # Human-readable description
    severity: str  # "low", "medium", "high"
    location: str  # Where in response (line/phrase)
    detected_value: Optional[str] = None  # The problematic value


@dataclass
class ValidationResult:
    """Result of response validation"""
    is_valid: bool  # Overall validity
    issues: List[ValidationIssue]  # List of detected issues
    severity: str  # Overall severity: "none", "low", "medium", "high"
    should_flag: bool  # True if response should be flagged for review
    confidence_adjustment: Optional[str] = None  # Suggest lowering confidence


class ResponseValidator:
    """
    Post-process LLM responses to detect hallucinations and quality issues
    """
    
    # Uncertainty patterns (phrases indicating LLM is unsure)
    UNCERTAINTY_PATTERNS = [
        # English
        r'\b(might|may|could|possibly|probably|perhaps|maybe)\b',
        r'\b(I think|I believe|I assume|it seems|appears to be)\b',
        r'\b(not sure|uncertain|unclear|not certain)\b',
        r'\b(should be|would be|could be)\b',
        r'\b(likely|unlikely|potential|possible)\b',
        # Turkish
        r'\b(olabilir|muhtemelen|belki|sanırım|gibi görünüyor)\b',
        r'\b(emin değilim|kesin değil|belirsiz)\b'
    ]
    
    # Forbidden phrases for wireless/battery when capability doesn't exist
    WIRELESS_FORBIDDEN = [
        r'\bwifi\b', r'\bwi-fi\b', r'\bwireless\b', r'\bnetwork\b',
        r'\baccess point\b', r'\bconnect unit\b', r'\bpairing\b'
    ]
    
    BATTERY_FORBIDDEN = [
        r'\bbattery\b', r'\bcharging\b', r'\bcharge\b', r'\bcharger\b'
    ]
    
    def __init__(
        self,
        max_uncertainty_count: int = 2,
        min_response_length: int = 30,
        flag_uncertain_responses: bool = True,
        verify_numbers: bool = True
    ):
        """
        Initialize validator
        
        Args:
            max_uncertainty_count: Max uncertain phrases before flagging
            min_response_length: Minimum chars for valid response
            flag_uncertain_responses: Whether to flag uncertain responses
            verify_numbers: Whether to verify numerical values
        """
        self.max_uncertainty_count = max_uncertainty_count
        self.min_response_length = min_response_length
        self.flag_uncertain_responses = flag_uncertain_responses
        self.verify_numbers = verify_numbers
        
        logger.info(f"ResponseValidator initialized: max_uncertainty={max_uncertainty_count}, min_length={min_response_length}")
    
    def validate_response(
        self,
        response: str,
        query: str,
        context: str,
        product_info: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Run all validation checks on LLM response
        
        Args:
            response: LLM-generated response text
            query: Original user query
            context: Context that was provided to LLM
            product_info: Optional product information dict
        
        Returns:
            ValidationResult with issues and severity
        """
        issues = []
        
        # Check 1: Uncertainty phrase detection
        if self.flag_uncertain_responses:
            uncertainty_issues = self._detect_uncertainty_phrases(response)
            issues.extend(uncertainty_issues)
        
        # Check 2: Response length
        length_issue = self._check_response_length(response)
        if length_issue:
            issues.append(length_issue)
        
        # Check 3: Numerical value verification
        if self.verify_numbers and context:
            number_issues = self._verify_numerical_values(response, context)
            issues.extend(number_issues)
        
        # Check 4: Product mismatch detection
        if product_info:
            mismatch_issue = self._check_product_mismatch(response, product_info)
            if mismatch_issue:
                issues.append(mismatch_issue)
            
            # Check 5: Forbidden content based on capabilities
            forbidden_issues = self._check_forbidden_content(response, product_info)
            issues.extend(forbidden_issues)
        
        # Determine overall severity
        severity = self._calculate_severity(issues)
        
        # Decide if should flag for review
        should_flag = (
            severity in ["high", "medium"] or
            len([i for i in issues if i.type == "uncertainty"]) > self.max_uncertainty_count
        )
        
        # Suggest confidence adjustment
        confidence_adjustment = None
        if len(issues) > 0 and severity in ["medium", "high"]:
            confidence_adjustment = "low"
        
        is_valid = len(issues) == 0 or severity == "low"
        
        logger.info(
            f"Validation complete: {len(issues)} issues, severity={severity}, "
            f"should_flag={should_flag}"
        )
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            severity=severity,
            should_flag=should_flag,
            confidence_adjustment=confidence_adjustment
        )
    
    def _detect_uncertainty_phrases(self, text: str) -> List[ValidationIssue]:
        """
        Detect phrases indicating LLM is unsure
        
        Args:
            text: Response text
        
        Returns:
            List of uncertainty issues
        """
        issues = []
        text_lower = text.lower()
        
        for pattern in self.UNCERTAINTY_PATTERNS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Get surrounding context (20 chars before/after)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                
                issues.append(ValidationIssue(
                    type="uncertainty",
                    description=f"Uncertain phrase detected: '{match.group()}'",
                    severity="low" if len(issues) <self.max_uncertainty_count else "medium",
                    location=f"char {match.start()}-{match.end()}",
                    detected_value=match.group()
                ))
        
        if issues:
            logger.debug(f"Detected {len(issues)} uncertainty phrases")
        
        return issues
    
    def _check_response_length(self, text: str) -> Optional[ValidationIssue]:
        """
        Check if response is too short (likely insufficient)
        
        Args:
            text: Response text
        
        Returns:
            Issue if too short, None otherwise
        """
        length = len(text.strip())
        
        if length < self.min_response_length:
            logger.warning(f"Response too short: {length} chars (min: {self.min_response_length})")
            return ValidationIssue(
                type="short_response",
                description=f"Response is only {length} characters (minimum: {self.min_response_length})",
                severity="high",
                location="entire response",
                detected_value=str(length)
            )
        
        return None
    
    def _verify_numerical_values(self, response: str, context: str) -> List[ValidationIssue]:
        """
        Verify all numbers in response exist in context
        
        Args:
            response: LLM response
            context: Provided context
        
        Returns:
            List of number mismatch issues
        """
        issues = []
        
        # Extract numbers with units from response
        response_numbers = self._extract_numbers_with_units(response)
        
        # Extract numbers with units from context
        context_numbers = self._extract_numbers_with_units(context)
        
        # Check each response number exists in context
        for number, unit, full_match in response_numbers:
            # Check if this exact number+unit appears in context
            found = False
            for ctx_num, ctx_unit, _ in context_numbers:
                if abs(number - ctx_num) < 0.01 and unit.lower() == ctx_unit.lower():
                    found = True
                    break
            
            if not found:
                logger.warning(f"Hallucinated number: {number} {unit} not in context")
                issues.append(ValidationIssue(
                    type="number_mismatch",
                    description=f"Number '{number} {unit}' not found in provided context",
                    severity="high",
                    location="numerical specification",
                    detected_value=f"{number} {unit}"
                ))
        
        return issues
    
    def _extract_numbers_with_units(self, text: str) -> List[Tuple[float, str, str]]:
        """
        Extract all numbers with units from text
        
        Returns:
            List of (number, unit, full_match) tuples
        """
        # Pattern: number + optional decimal + unit
        # Units: Nm, kg, mm, bar, rpm, V, A, W, etc.
        pattern = r'(\d+(?:\.\d+)?)\s*(Nm|nm|kg|g|mm|cm|m|bar|psi|rpm|RPM|V|A|W|Hz|°C|°F)'
        
        matches = []
        for match in re.finditer(pattern, text):
            try:
                number = float(match.group(1))
                unit = match.group(2)
                matches.append((number, unit, match.group(0)))
            except ValueError:
                continue
        
        return matches
    
    def _check_product_mismatch(self, response: str, product_info: Dict) -> Optional[ValidationIssue]:
        """
        Detect if response mentions different product than queried
        
        Args:
            response: LLM response
            product_info: Product information
        
        Returns:
            Issue if mismatch detected, None otherwise
        """
        actual_product = product_info.get('model_name', '').upper()
        
        # Extract product series from actual product (e.g., "EPB" from "EPB8-1800-4Q")
        actual_series = re.match(r'^([A-Z]+)', actual_product)
        if not actual_series:
            return None
        
        actual_series = actual_series.group(1)
        
        # Known product series to check for
        product_series = ['EPB', 'EPBC', 'EAD', 'EPD', 'EFD', 'EIDS', 'ERS', 'ECS', 
                         'CVI3', 'CVIC', 'CVIR', 'CVIL', 'AXON', 'CONNECT']
        
        # Check if response mentions other product series
        response_upper = response.upper()
        for series in product_series:
            if series != actual_series and series in response_upper:
                logger.warning(f"Product mismatch: Response mentions {series} but query is for {actual_series}")
                return ValidationIssue(
                    type="product_mismatch",
                    description=f"Response mentions '{series}' but query is about '{actual_series}'",
                    severity="medium",
                    location="product reference",
                    detected_value=series
                )
        
        return None
    
    def _check_forbidden_content(self, response: str, product_info: Dict) -> List[ValidationIssue]:
        """
        Check for suggestions outside product capabilities
        
        Args:
            response: LLM response
            product_info: Product information with capabilities
        
        Returns:
            List of forbidden content issues
        """
        issues = []
        response_lower = response.lower()
        
        # Check WiFi suggestions for non-WiFi products
        if not product_info.get('wireless', False):
            for pattern in self.WIRELESS_FORBIDDEN:
                if re.search(pattern, response_lower):
                    issues.append(ValidationIssue(
                        type="forbidden_content",
                        description="Response suggests WiFi troubleshooting but product has no WiFi capability",
                        severity="high",
                        location="wireless suggestion",
                        detected_value=pattern
                    ))
                    break  # One issue per type is enough
        
        # Check battery suggestions for non-battery products
        if not product_info.get('battery_powered', False):
            for pattern in self.BATTERY_FORBIDDEN:
                if re.search(pattern, response_lower):
                    issues.append(ValidationIssue(
                        type="forbidden_content",
                        description="Response suggests battery troubleshooting but product is not battery-powered",
                        severity="high",
                        location="battery suggestion",
                        detected_value=pattern
                    ))
                    break
        
        return issues
    
    def _calculate_severity(self, issues: List[ValidationIssue]) -> str:
        """
        Calculate overall severity from individual issues
        
        Args:
            issues: List of validation issues
        
        Returns:
            Overall severity: "none", "low", "medium", "high"
        """
        if not issues:
            return "none"
        
        # Count by severity
        high_count = sum(1 for i in issues if i.severity == "high")
        medium_count = sum(1 for i in issues if i.severity == "medium")
        low_count = sum(1 for i in issues if i.severity == "low")
        
        # Overall severity logic
        if high_count > 0:
            return "high"
        elif medium_count >= 2:
            return "high"
        elif medium_count > 0:
            return "medium"
        elif low_count > 3:
            return "medium"
        else:
            return "low"
