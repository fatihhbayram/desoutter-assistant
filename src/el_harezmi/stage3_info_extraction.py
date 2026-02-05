"""
Stage 3: Information Extraction

Structured information extraction from retrieved chunks using LLM.

Responsibilities:
- Extract structured data from chunks based on intent type
- Use intent-specific extraction templates
- Validate extracted JSON against schemas
- Handle multi-source information merging
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from .stage1_intent_classifier import IntentType, IntentResult
from .stage2_retrieval_strategy import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Prerequisites:
    """Prerequisites for a procedure"""
    controller: Optional[Dict[str, str]] = None  # {"model": "CVI3", "min_version": "2.5"}
    firmware: Optional[Dict[str, str]] = None  # {"tool_min": "1.8", "controller_min": "2.5"}
    accessories: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProcedureStep:
    """A single step in a procedure"""
    step_number: int
    action: str
    details: Optional[str] = None
    warning: Optional[str] = None
    expected_result: Optional[str] = None


@dataclass
class ParameterRange:
    """Parameter range specification"""
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: Optional[str] = None
    default_value: Optional[float] = None


@dataclass
class CompatibilityInfo:
    """Compatibility information"""
    compatible_controllers: List[Dict[str, Any]] = field(default_factory=list)
    firmware_requirements: Optional[Dict[str, str]] = None
    compatible_accessories: Dict[str, List[str]] = field(default_factory=dict)
    incompatible_items: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TroubleshootInfo:
    """Troubleshooting information"""
    problem_description: str = ""
    possible_causes: List[str] = field(default_factory=list)
    diagnostic_steps: List[str] = field(default_factory=list)
    solutions: List[ProcedureStep] = field(default_factory=list)
    esde_reference: Optional[str] = None
    esde_description: Optional[str] = None
    affected_models: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Result of information extraction"""
    intent: IntentType
    product_model: Optional[str] = None
    
    # Configuration extraction
    prerequisites: Optional[Prerequisites] = None
    procedure: List[ProcedureStep] = field(default_factory=list)
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    
    # Compatibility extraction
    compatibility: Optional[CompatibilityInfo] = None
    
    # Troubleshooting extraction
    troubleshoot: Optional[TroubleshootInfo] = None
    
    # Specification extraction
    specifications: Dict[str, Any] = field(default_factory=dict)
    
    # General fields
    warnings: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    raw_extraction: Optional[Dict[str, Any]] = None


class InfoExtractor:
    """
    Stage 3: Structured Information Extractor
    
    Uses LLM to extract structured information from retrieved chunks.
    """
    
    # Extraction templates for each intent type
    EXTRACTION_TEMPLATES = {
        IntentType.CONFIGURATION: {
            "extract": [
                "prerequisites",
                "step_by_step_procedure",
                "parameter_ranges",
                "warnings",
                "verification_steps"
            ],
            "output_format": "structured_procedure",
            "prompt_template": """
From the following technical documents, extract CONFIGURATION PROCEDURE for:
Product: {product_model}
Task: {task_description}
Target Parameter: {parameter_type} = {target_value}

Extract in this JSON structure:
{{
  "prerequisites": {{
    "controller": {{"model": "string", "min_version": "string"}},
    "firmware": {{"tool_min": "string", "controller_min": "string"}},
    "accessories": ["string"]
  }},
  "procedure": [
    {{"step": 1, "action": "string", "details": "string"}}
  ],
  "parameter_ranges": [
    {{"parameter": "string", "min": number, "max": number, "unit": "string"}}
  ],
  "warnings": ["string"],
  "verification": ["string"]
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Do NOT invent steps.
If information is not available, use null or empty arrays.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.COMPATIBILITY: {
            "extract": [
                "compatible_controllers",
                "firmware_requirements",
                "compatible_accessories",
                "incompatible_items",
                "recommendations"
            ],
            "output_format": "compatibility_matrix",
            "prompt_template": """
From the following technical documents, extract COMPATIBILITY INFORMATION for:
Product: {product_model}
Query: {task_description}

Extract in this JSON structure:
{{
  "compatible_controllers": [
    {{"model": "string", "min_version": "string", "max_version": "string or null", "recommended": boolean}}
  ],
  "firmware_requirements": {{
    "tool_min": "string",
    "controller_min": "string"
  }},
  "compatible_accessories": {{
    "docks": ["string"],
    "batteries": ["string"],
    "cables": ["string"]
  }},
  "incompatible_items": ["string"],
  "recommendations": ["string"]
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Do NOT guess compatibility.
If information is not available, use null or empty arrays.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.TROUBLESHOOT: {
            "extract": [
                "problem_description",
                "possible_causes",
                "diagnostic_steps",
                "solutions",
                "esde_reference"
            ],
            "output_format": "troubleshooting_guide",
            "prompt_template": """
From the following technical documents, extract TROUBLESHOOTING INFORMATION for:
Product: {product_model}
Problem: {task_description}
Error Code: {error_code}

Extract in this JSON structure:
{{
  "problem_description": "string",
  "possible_causes": ["string"],
  "diagnostic_steps": ["string"],
  "solutions": [
    {{"step": 1, "action": "string", "expected_result": "string"}}
  ],
  "esde_reference": "string or null",
  "esde_description": "string or null",
  "affected_models": ["string"]
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Do NOT invent solutions.
If this is a known manufacturing defect (ESDE), include the ESDE code.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.ERROR_CODE: {
            "extract": [
                "error_code",
                "error_description",
                "causes",
                "solutions"
            ],
            "output_format": "error_code_info",
            "prompt_template": """
From the following technical documents, extract ERROR CODE INFORMATION for:
Error Code: {error_code}
Product: {product_model}

Extract in this JSON structure:
{{
  "error_code": "string",
  "error_description": "string",
  "possible_causes": ["string"],
  "diagnostic_steps": ["string"],
  "solutions": [
    {{"step": 1, "action": "string", "expected_result": "string"}}
  ],
  "related_esde": "string or null"
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Do NOT invent error meanings.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.SPECIFICATION: {
            "extract": [
                "specifications",
                "dimensions",
                "capabilities"
            ],
            "output_format": "spec_sheet",
            "prompt_template": """
From the following technical documents, extract TECHNICAL SPECIFICATIONS for:
Product: {product_model}
Query: {task_description}

Extract in this JSON structure:
{{
  "model": "string",
  "specifications": {{
    "torque_range": {{"min": number, "max": number, "unit": "string"}},
    "speed_range": {{"min": number, "max": number, "unit": "string"}},
    "weight": {{"value": number, "unit": "string"}},
    "voltage": "string",
    "features": ["string"]
  }},
  "dimensions": {{
    "length": "string",
    "width": "string",
    "height": "string"
  }},
  "capabilities": ["string"]
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Use null for missing values.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.PROCEDURE: {
            "extract": [
                "prerequisites",
                "procedure_steps",
                "warnings",
                "verification"
            ],
            "output_format": "structured_procedure",
            "prompt_template": """
From the following technical documents, extract PROCEDURE for:
Product: {product_model}
Task: {task_description}

Extract in this JSON structure:
{{
  "title": "string",
  "prerequisites": {{
    "tools": ["string"],
    "warnings": ["string"]
  }},
  "procedure": [
    {{"step": 1, "action": "string", "details": "string", "warning": "string or null"}}
  ],
  "verification": ["string"],
  "estimated_time": "string or null"
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Preserve step order exactly.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.CAPABILITY_QUERY: {
            "extract": [
                "capability",
                "supported",
                "details"
            ],
            "output_format": "capability_info",
            "prompt_template": """
From the following technical documents, extract CAPABILITY INFORMATION for:
Product: {product_model}
Query: {task_description}

Extract in this JSON structure:
{{
  "capability_queried": "string",
  "supported": boolean,
  "details": "string",
  "limitations": ["string"],
  "related_features": ["string"]
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Do NOT guess capabilities.
Return ONLY valid JSON, no explanations.
""",
        },
        
        IntentType.ACCESSORY_QUERY: {
            "extract": [
                "compatible_accessories",
                "recommendations"
            ],
            "output_format": "accessory_info",
            "prompt_template": """
From the following technical documents, extract ACCESSORY INFORMATION for:
Product: {product_model}
Query: {task_description}

Extract in this JSON structure:
{{
  "accessory_type": "string",
  "compatible_accessories": [
    {{"name": "string", "model": "string", "notes": "string"}}
  ],
  "incompatible": ["string"],
  "recommendations": ["string"]
}}

Documents:
{chunks}

CRITICAL: Extract ONLY from provided documents. Do NOT guess compatibility.
Return ONLY valid JSON, no explanations.
""",
        },
    }
    
    # Default template for unmapped intents
    DEFAULT_TEMPLATE = {
        "extract": ["content"],
        "output_format": "general",
        "prompt_template": """
From the following technical documents, extract relevant information for:
Product: {product_model}
Query: {task_description}

Extract in this JSON structure:
{{
  "summary": "string",
  "key_points": ["string"],
  "related_topics": ["string"]
}}

Documents:
{chunks}

Return ONLY valid JSON, no explanations.
""",
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize Info Extractor.
        
        Args:
            llm_client: LLM client for extraction (Ollama/OpenAI compatible)
        """
        self.llm_client = llm_client
    
    def get_template(self, intent: IntentType) -> Dict[str, Any]:
        """Get extraction template for an intent type"""
        return self.EXTRACTION_TEMPLATES.get(intent, self.DEFAULT_TEMPLATE)
    
    async def extract(
        self,
        intent_result: IntentResult,
        retrieval_result: RetrievalResult
    ) -> ExtractionResult:
        """
        Extract structured information from retrieved chunks.
        
        Args:
            intent_result: Result from Stage 1
            retrieval_result: Result from Stage 2
            
        Returns:
            ExtractionResult with structured data
        """
        template = self.get_template(intent_result.primary_intent)
        
        # Prepare chunk content
        chunk_content = self._prepare_chunks(retrieval_result.chunks)
        
        # Build extraction prompt
        prompt = self._build_prompt(
            template=template,
            intent_result=intent_result,
            chunks=chunk_content
        )
        
        logger.info(f"Extracting info for intent: {intent_result.primary_intent.value}")
        
        # Execute LLM extraction
        raw_extraction = await self._execute_extraction(prompt)
        
        # Check if extraction is empty or failed
        if self._is_empty_extraction(raw_extraction):
            logger.warning("LLM extraction returned empty - using pattern-based fallback")
            return self.extract_without_llm(intent_result, retrieval_result)
        
        # Parse and validate extraction
        result = self._parse_extraction(
            raw_extraction=raw_extraction,
            intent=intent_result.primary_intent,
            intent_result=intent_result,
            retrieval_result=retrieval_result
        )
        
        return result
    
    def _is_empty_extraction(self, extraction: Dict[str, Any]) -> bool:
        """
        Check if extraction result is empty or contains only null/empty values.
        
        Returns True if extraction should be considered failed.
        """
        if not extraction:
            return True
        
        # Check if all values are None, empty strings, or empty lists
        for key, value in extraction.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return False
            if isinstance(value, (list, dict)) and len(value) > 0:
                return False
            if isinstance(value, (int, float)) and value != 0:
                return False
        
        return True
    
    def _prepare_chunks(self, chunks: List[RetrievedChunk]) -> str:
        """Prepare chunk content for LLM prompt"""
        chunk_texts = []
        
        for i, chunk in enumerate(chunks[:10]):  # Limit to top 10 chunks
            header = f"[Document {i+1}]"
            if chunk.document_type:
                header += f" Type: {chunk.document_type}"
            if chunk.product_model:
                header += f" | Product: {chunk.product_model}"
            if chunk.section_hierarchy:
                header += f" | Section: {chunk.section_hierarchy}"
            
            chunk_texts.append(f"{header}\n{chunk.content}\n")
        
        return "\n---\n".join(chunk_texts)
    
    def _build_prompt(
        self,
        template: Dict[str, Any],
        intent_result: IntentResult,
        chunks: str
    ) -> str:
        """Build extraction prompt from template"""
        entities = intent_result.entities
        
        prompt = template["prompt_template"].format(
            product_model=entities.product_model or "Not specified",
            task_description=intent_result.raw_query,
            parameter_type=entities.parameter_type or "Not specified",
            target_value=entities.target_value or "Not specified",
            error_code=entities.error_code or "Not specified",
            chunks=chunks
        )
        
        return prompt
    
    async def _execute_extraction(self, prompt: str) -> Dict[str, Any]:
        """Execute LLM extraction"""
        if not self.llm_client:
            logger.warning("No LLM client configured - returning empty extraction")
            return {}
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for structured extraction
                max_tokens=2000
            )
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning("No JSON found in LLM response")
                return {}
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {}
    
    def _parse_extraction(
        self,
        raw_extraction: Dict[str, Any],
        intent: IntentType,
        intent_result: IntentResult,
        retrieval_result: RetrievalResult
    ) -> ExtractionResult:
        """Parse raw extraction into ExtractionResult"""
        
        result = ExtractionResult(
            intent=intent,
            product_model=intent_result.entities.product_model,
            sources=[c.chunk_id for c in retrieval_result.chunks[:5]],
            raw_extraction=raw_extraction,
            confidence=0.8 if raw_extraction else 0.0
        )
        
        if not raw_extraction:
            return result
        
        # Parse based on intent type
        if intent == IntentType.CONFIGURATION:
            result = self._parse_configuration(result, raw_extraction)
        elif intent == IntentType.COMPATIBILITY:
            result = self._parse_compatibility(result, raw_extraction)
        elif intent in [IntentType.TROUBLESHOOT, IntentType.ERROR_CODE]:
            result = self._parse_troubleshoot(result, raw_extraction)
        elif intent == IntentType.SPECIFICATION:
            result = self._parse_specification(result, raw_extraction)
        elif intent == IntentType.PROCEDURE:
            result = self._parse_procedure(result, raw_extraction)
        
        # Extract common fields
        result.warnings = raw_extraction.get("warnings", [])
        
        return result
    
    def _parse_configuration(
        self,
        result: ExtractionResult,
        raw: Dict[str, Any]
    ) -> ExtractionResult:
        """Parse configuration extraction"""
        
        # Parse prerequisites
        prereq_data = raw.get("prerequisites", {})
        if prereq_data:
            result.prerequisites = Prerequisites(
                controller=prereq_data.get("controller"),
                firmware=prereq_data.get("firmware"),
                accessories=prereq_data.get("accessories", []),
                warnings=prereq_data.get("warnings", [])
            )
        
        # Parse procedure steps
        procedure_data = raw.get("procedure", [])
        for step in procedure_data:
            result.procedure.append(ProcedureStep(
                step_number=step.get("step", 0),
                action=step.get("action", ""),
                details=step.get("details"),
                warning=step.get("warning"),
                expected_result=step.get("expected_result")
            ))
        
        # Parse parameter ranges
        ranges_data = raw.get("parameter_ranges", [])
        for param in ranges_data:
            result.parameter_ranges.append(ParameterRange(
                parameter=param.get("parameter", ""),
                min_value=param.get("min"),
                max_value=param.get("max"),
                unit=param.get("unit")
            ))
        
        # Parse verification
        result.verification_steps = raw.get("verification", [])
        
        return result
    
    def _parse_compatibility(
        self,
        result: ExtractionResult,
        raw: Dict[str, Any]
    ) -> ExtractionResult:
        """Parse compatibility extraction"""
        
        result.compatibility = CompatibilityInfo(
            compatible_controllers=raw.get("compatible_controllers", []),
            firmware_requirements=raw.get("firmware_requirements"),
            compatible_accessories=raw.get("compatible_accessories", {}),
            incompatible_items=raw.get("incompatible_items", []),
            recommendations=raw.get("recommendations", [])
        )
        
        return result
    
    def _parse_troubleshoot(
        self,
        result: ExtractionResult,
        raw: Dict[str, Any]
    ) -> ExtractionResult:
        """Parse troubleshooting extraction"""
        
        solutions = []
        for step in raw.get("solutions", []):
            solutions.append(ProcedureStep(
                step_number=step.get("step", 0),
                action=step.get("action", ""),
                expected_result=step.get("expected_result")
            ))
        
        result.troubleshoot = TroubleshootInfo(
            problem_description=raw.get("problem_description", ""),
            possible_causes=raw.get("possible_causes", []),
            diagnostic_steps=raw.get("diagnostic_steps", []),
            solutions=solutions,
            esde_reference=raw.get("esde_reference"),
            esde_description=raw.get("esde_description"),
            affected_models=raw.get("affected_models", [])
        )
        
        return result
    
    def _parse_specification(
        self,
        result: ExtractionResult,
        raw: Dict[str, Any]
    ) -> ExtractionResult:
        """Parse specification extraction"""
        
        result.specifications = raw.get("specifications", {})
        
        return result
    
    def _parse_procedure(
        self,
        result: ExtractionResult,
        raw: Dict[str, Any]
    ) -> ExtractionResult:
        """Parse procedure extraction"""
        
        # Parse prerequisites
        prereq_data = raw.get("prerequisites", {})
        if prereq_data:
            result.prerequisites = Prerequisites(
                accessories=prereq_data.get("tools", []),
                warnings=prereq_data.get("warnings", [])
            )
        
        # Parse procedure steps
        procedure_data = raw.get("procedure", [])
        for step in procedure_data:
            result.procedure.append(ProcedureStep(
                step_number=step.get("step", 0),
                action=step.get("action", ""),
                details=step.get("details"),
                warning=step.get("warning")
            ))
        
        # Parse verification
        result.verification_steps = raw.get("verification", [])
        
        return result
    
    def extract_without_llm(
        self,
        intent_result: IntentResult,
        retrieval_result: RetrievalResult
    ) -> ExtractionResult:
        """
        Fallback extraction without LLM using pattern matching.
        
        Used when:
        - LLM is unavailable
        - LLM extraction returns empty/null values
        - Simple queries that don't require LLM
        
        Implements intent-specific pattern extraction for each intent type.
        """
        result = ExtractionResult(
            intent=intent_result.primary_intent,
            product_model=intent_result.entities.product_model,
            sources=[c.chunk_id for c in retrieval_result.chunks[:5]],
            confidence=0.6
        )
        
        # Combine all chunk content for pattern matching
        all_content = " ".join([c.content for c in retrieval_result.chunks])
        
        # Extract common fields (warnings)
        result.warnings = self._extract_warnings(all_content)
        
        # Intent-specific extraction
        intent = intent_result.primary_intent
        
        if intent == IntentType.CONFIGURATION:
            self._extract_configuration_patterns(result, all_content, retrieval_result)
        elif intent == IntentType.COMPATIBILITY:
            self._extract_compatibility_patterns(result, all_content, retrieval_result)
        elif intent in [IntentType.TROUBLESHOOT, IntentType.ERROR_CODE]:
            self._extract_troubleshoot_patterns(result, all_content, intent_result)
        elif intent == IntentType.SPECIFICATION:
            self._extract_specification_patterns(result, all_content, retrieval_result)
        elif intent in [IntentType.PROCEDURE, IntentType.HOW_TO, IntentType.INSTALLATION,
                        IntentType.FIRMWARE, IntentType.CALIBRATION]:
            self._extract_procedure_patterns(result, all_content)
        elif intent == IntentType.CAPABILITY_QUERY:
            self._extract_capability_patterns(result, all_content)
        elif intent == IntentType.ACCESSORY_QUERY:
            self._extract_accessory_patterns(result, all_content)
        
        # Set confidence based on extraction quality
        result.confidence = self._calculate_extraction_confidence(result)
        
        return result
    
    def _extract_warnings(self, content: str) -> List[str]:
        """Extract warning/caution statements from content"""
        warnings = []
        warning_patterns = [
            r"(?:WARNING|CAUTION|UYARI|DİKKAT)[:\s]+([^.!]+[.!])",
            r"⚠️\s*([^.!]+[.!])",
            r"(?:IMPORTANT|ÖNEMLİ)[:\s]+([^.!]+[.!])",
        ]
        for pattern in warning_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            warnings.extend([m.strip() for m in matches if len(m.strip()) > 10])
        return warnings[:5]  # Limit to 5 warnings
    
    def _extract_configuration_patterns(
        self,
        result: ExtractionResult,
        content: str,
        retrieval_result: RetrievalResult
    ) -> None:
        """Extract configuration-related patterns"""
        # Extract parameter ranges from tables or text
        # Pattern: "Torque: 5-85 Nm" or "Max Torque: 85 Nm"
        param_patterns = [
            r"(Torque|Tork)[:\s]*([\d.]+)\s*[-–]\s*([\d.]+)\s*(Nm|N\.m)",
            r"(Angle|Açı)[:\s]*([\d.]+)\s*[-–]\s*([\d.]+)\s*(°|deg|derece)",
            r"(Speed|Hız)[:\s]*([\d.]+)\s*[-–]\s*([\d.]+)\s*(RPM|rpm)",
            r"Max\.?\s*(Torque|Tork)[:\s]*([\d.]+)\s*(Nm|N\.m)",
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 4:  # Range pattern
                    result.parameter_ranges.append(ParameterRange(
                        parameter=match[0],
                        min_value=float(match[1]),
                        max_value=float(match[2]),
                        unit=match[3]
                    ))
                elif len(match) >= 3:  # Max only pattern
                    result.parameter_ranges.append(ParameterRange(
                        parameter=match[0],
                        max_value=float(match[1]),
                        unit=match[2]
                    ))
        
        # Extract controller requirements
        controller_patterns = [
            r"(CVI3|CVIR|CVIC-II|CONNECT)\s*v?([\d.]+)\+?",
            r"Controller[:\s]*(CVI3|CVIR|CVIC-II|CONNECT)",
        ]
        for pattern in controller_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                result.prerequisites = Prerequisites(
                    controller={"model": matches[0][0].upper() if isinstance(matches[0], tuple) else matches[0]}
                )
                break
        
        # Extract procedure steps
        self._extract_numbered_steps(result, content)
    
    def _extract_compatibility_patterns(
        self,
        result: ExtractionResult,
        content: str,
        retrieval_result: RetrievalResult
    ) -> None:
        """Extract compatibility information from content"""
        compatible_controllers = []
        
        # Pattern: "Compatible with CVI3 v2.5+"
        compat_patterns = [
            r"(?:Compatible|Uyumlu)[^.]*?(CVI3|CVIR|CVIC-II|CONNECT)\s*v?([\d.]+)?",
            r"(CVI3|CVIR|CVIC-II|CONNECT)\s*v?([\d.]+)?\s*(?:or later|veya üstü|\+)",
            r"Requires?\s*(CVI3|CVIR|CVIC-II|CONNECT)\s*v?([\d.]+)?",
        ]
        
        for pattern in compat_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                controller = match[0].upper() if match[0] else None
                version = match[1] if len(match) > 1 else None
                if controller:
                    compatible_controllers.append({
                        "model": controller,
                        "min_version": version,
                        "recommended": True
                    })
        
        # Extract firmware requirements
        firmware_req = {}
        fw_patterns = [
            r"Firmware[:\s]*v?([\d.]+)\+?",
            r"Min\.?\s*firmware[:\s]*v?([\d.]+)",
        ]
        for pattern in fw_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                firmware_req["min_version"] = match.group(1)
                break
        
        # Extract compatible accessories
        accessories = {"docks": [], "batteries": [], "cables": []}
        accessory_patterns = [
            (r"(?:Dock|Dok)[:\s]*([A-Z0-9-]+)", "docks"),
            (r"(?:Battery|Batarya)[:\s]*([A-Z0-9-]+)", "batteries"),
            (r"(?:Cable|Kablo)[:\s]*(USB-C|USB|WiFi|Ethernet)", "cables"),
        ]
        for pattern, category in accessory_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            accessories[category].extend(matches)
        
        result.compatibility = CompatibilityInfo(
            compatible_controllers=compatible_controllers,
            firmware_requirements=firmware_req if firmware_req else None,
            compatible_accessories=accessories
        )
    
    def _extract_troubleshoot_patterns(
        self,
        result: ExtractionResult,
        content: str,
        intent_result: IntentResult
    ) -> None:
        """Extract troubleshooting information"""
        # Extract ESDE reference
        esde_pattern = r"(ESDE-\d{4,5})"
        esde_match = re.search(esde_pattern, content)
        esde_ref = esde_match.group(1) if esde_match else None
        
        # Extract error code if present in query
        error_code = intent_result.entities.error_code
        
        # Extract possible causes
        causes = []
        cause_patterns = [
            r"(?:Cause|Sebep|Neden)[:\s]*([^.!]+[.!])",
            r"(?:caused by|nedeniyle)[:\s]*([^.!]+[.!])",
        ]
        for pattern in cause_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            causes.extend([m.strip() for m in matches])
        
        # Extract solutions
        solutions = []
        solution_patterns = [
            r"(?:Solution|Çözüm)[:\s]*([^.!]+[.!])",
            r"(?:Fix|Düzeltme)[:\s]*([^.!]+[.!])",
        ]
        for pattern in solution_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for i, sol in enumerate(matches[:5]):
                solutions.append(ProcedureStep(
                    step_number=i + 1,
                    action=sol.strip()
                ))
        
        # If no explicit solutions, extract numbered steps
        if not solutions:
            self._extract_numbered_steps(result, content)
            solutions = result.procedure
        
        result.troubleshoot = TroubleshootInfo(
            problem_description=intent_result.raw_query,
            possible_causes=causes[:5],
            solutions=solutions,
            esde_reference=esde_ref,
            affected_models=[intent_result.entities.product_model] if intent_result.entities.product_model else []
        )
    
    def _extract_specification_patterns(
        self,
        result: ExtractionResult,
        content: str,
        retrieval_result: RetrievalResult
    ) -> None:
        """Extract technical specifications from content (tables, lists)"""
        specs = {}
        
        # Common specification patterns
        spec_patterns = [
            (r"(?:Max\.?\s*)?Torque[:\s]*([\d.]+)\s*(Nm|N\.m)", "max_torque"),
            (r"(?:Max\.?\s*)?Speed[:\s]*([\d.]+)\s*(RPM|rpm)", "max_speed"),
            (r"Weight[:\s]*([\d.]+)\s*(kg|g)", "weight"),
            (r"Length[:\s]*([\d.]+)\s*(mm|cm)", "length"),
            (r"Voltage[:\s]*([\d.]+)\s*(V|volt)", "voltage"),
            (r"Battery[:\s]*([\d.]+)\s*(Ah|mAh)", "battery_capacity"),
        ]
        
        for pattern, spec_name in spec_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                specs[spec_name] = f"{match.group(1)} {match.group(2)}"
        
        # Extract table data if present in chunks
        for chunk in retrieval_result.chunks:
            if chunk.chunk_type == "table_row" and chunk.metadata:
                # Table rows often have key-value pairs in metadata
                for key, value in chunk.metadata.items():
                    if key not in ["source", "chunk_id", "score"]:
                        specs[key] = value
        
        result.specifications = specs
    
    def _extract_procedure_patterns(self, result: ExtractionResult, content: str) -> None:
        """Extract procedure steps from content"""
        self._extract_numbered_steps(result, content)
        
        # Extract prerequisites/tools needed
        tool_patterns = [
            r"(?:Required|Gerekli)[:\s]*([^.]+\.)",
            r"(?:Tools|Aletler)[:\s]*([^.]+\.)",
            r"(?:Prerequisites|Ön koşullar)[:\s]*([^.]+\.)",
        ]
        accessories = []
        for pattern in tool_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            accessories.extend([m.strip() for m in matches])
        
        if accessories:
            result.prerequisites = Prerequisites(accessories=accessories)
    
    def _extract_capability_patterns(self, result: ExtractionResult, content: str) -> None:
        """Extract capability information (WiFi, Bluetooth, etc.)"""
        capabilities = {}
        
        # Check for common capabilities
        cap_patterns = [
            (r"WiFi[:\s]*(Yes|No|Evet|Hayır|✓|✗|supported|destekleniyor)", "wifi"),
            (r"Bluetooth[:\s]*(Yes|No|Evet|Hayır|✓|✗|supported|destekleniyor)", "bluetooth"),
            (r"Display[:\s]*(LCD|LED|OLED|Yes|No)", "display"),
            (r"Data\s*Log(?:ging)?[:\s]*(Yes|No|Evet|Hayır|✓|✗)", "data_logging"),
        ]
        
        for pattern, cap_name in cap_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1)
                # Normalize boolean values
                if value.lower() in ["yes", "evet", "✓", "supported", "destekleniyor"]:
                    capabilities[cap_name] = True
                elif value.lower() in ["no", "hayır", "✗"]:
                    capabilities[cap_name] = False
                else:
                    capabilities[cap_name] = value
        
        result.specifications = capabilities
    
    def _extract_accessory_patterns(self, result: ExtractionResult, content: str) -> None:
        """Extract accessory information"""
        accessories = {"docks": [], "batteries": [], "cables": [], "adapters": []}
        
        # Extract accessory names/models
        accessory_patterns = [
            (r"(?:Dock|Dok)[:\s]*([A-Z0-9][A-Z0-9-]+)", "docks"),
            (r"(?:Battery|Batarya|BAT)[:\s-]*([A-Z0-9][A-Z0-9-]+)", "batteries"),
            (r"(?:Adapter|Adaptör)[:\s]*([A-Z0-9][A-Z0-9-]+)", "adapters"),
            (r"(USB-C|USB-A|Ethernet|WiFi)\s*(?:cable|kablo)?", "cables"),
        ]
        
        for pattern, category in accessory_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            accessories[category].extend([m.upper() for m in matches if m])
        
        result.compatibility = CompatibilityInfo(
            compatible_accessories=accessories
        )
    
    def _extract_numbered_steps(self, result: ExtractionResult, content: str) -> None:
        """Extract numbered procedure steps"""
        # Multiple step patterns for different formats
        step_patterns = [
            r"(\d+)[\.\)\-]\s+([^.!\n]+[.!])",  # "1. Step text."
            r"Step\s*(\d+)[:\s]+([^.!\n]+[.!])",  # "Step 1: Step text."
            r"Adım\s*(\d+)[:\s]+([^.!\n]+[.!])",  # "Adım 1: Step text." (Turkish)
        ]
        
        all_steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            all_steps.extend(matches)
        
        # Deduplicate and sort by step number
        seen_numbers = set()
        for num, action in all_steps:
            step_num = int(num)
            if step_num not in seen_numbers and step_num <= 20:  # Limit to 20 steps
                seen_numbers.add(step_num)
                result.procedure.append(ProcedureStep(
                    step_number=step_num,
                    action=action.strip()
                ))
        
        # Sort by step number
        result.procedure.sort(key=lambda x: x.step_number)
    
    def _calculate_extraction_confidence(self, result: ExtractionResult) -> float:
        """
        Calculate confidence score based on extraction quality.
        
        Higher confidence if more fields are populated.
        """
        score = 0.3  # Base score for fallback extraction
        
        # Add points for populated fields
        if result.procedure:
            score += 0.15
        if result.parameter_ranges:
            score += 0.1
        if result.prerequisites:
            score += 0.1
        if result.compatibility and result.compatibility.compatible_controllers:
            score += 0.15
        if result.troubleshoot and result.troubleshoot.possible_causes:
            score += 0.1
        if result.specifications:
            score += 0.1
        if result.warnings:
            score += 0.05
        
        return min(score, 0.85)  # Cap at 0.85 for fallback (LLM can reach 0.95)
