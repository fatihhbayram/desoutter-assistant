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
        
        # Parse and validate extraction
        result = self._parse_extraction(
            raw_extraction=raw_extraction,
            intent=intent_result.primary_intent,
            intent_result=intent_result,
            retrieval_result=retrieval_result
        )
        
        return result
    
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
        
        Used when LLM is unavailable or for simple queries.
        """
        result = ExtractionResult(
            intent=intent_result.primary_intent,
            product_model=intent_result.entities.product_model,
            sources=[c.chunk_id for c in retrieval_result.chunks[:5]],
            confidence=0.6
        )
        
        # Simple pattern-based extraction
        all_content = " ".join([c.content for c in retrieval_result.chunks])
        
        # Extract warnings
        warning_patterns = [
            r"(?:WARNING|CAUTION|UYARI|DİKKAT)[:\s]+([^.]+\.)",
            r"⚠️\s*([^.]+\.)",
        ]
        for pattern in warning_patterns:
            matches = re.findall(pattern, all_content, re.IGNORECASE)
            result.warnings.extend(matches)
        
        # Extract numbered steps
        step_pattern = r"(\d+)\.\s+([^.]+\.)"
        steps = re.findall(step_pattern, all_content)
        for num, action in steps[:10]:  # Limit to 10 steps
            result.procedure.append(ProcedureStep(
                step_number=int(num),
                action=action.strip()
            ))
        
        return result
