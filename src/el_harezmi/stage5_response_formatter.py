"""
Stage 5: Response Formatter

Formats validated information into responses using intent-specific templates.
Supports both Turkish and English languages.

Responsibilities:
- Generate structured responses
- Apply intent-specific templates
- Include source citations
- Add warnings and recommendations
- Multi-language support (TR/EN)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

from .stage1_intent_classifier import IntentType, IntentResult
from .stage3_info_extraction import ExtractionResult, ProcedureStep
from .stage4_kg_validation import ValidationResult, ValidationStatus, ValidationIssue

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages"""
    TURKISH = "tr"
    ENGLISH = "en"


@dataclass
class FormattedResponse:
    """Final formatted response"""
    content: str
    intent: IntentType
    product_model: Optional[str]
    confidence: float
    sources: List[str]
    warnings: List[str]
    validation_status: ValidationStatus
    language: Language = Language.TURKISH
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseFormatter:
    """
    Stage 5: Response Formatter
    
    Generates structured responses based on validated extraction results.
    Supports Turkish and English languages.
    """
    
    # Turkish templates
    TEMPLATES_TR = {
        IntentType.CONFIGURATION: """
**{product_model} - {task_title}**

{prerequisites_section}

📋 **Adımlar:**
{procedure_section}

{parameter_section}

{warnings_section}

{verification_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.COMPATIBILITY: """
**{product_model} Uyumluluk Bilgileri**

✅ **Uyumlu Controller'lar:**
{controllers_table}

⚙️ **Firmware Gereksinimleri:**
{firmware_section}

🔌 **Uyumlu Aksesuarlar:**
{accessories_section}

{incompatible_section}

{recommendations_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.TROUBLESHOOT: """
**{product_model} - {problem_title}**

🔍 **Olası Nedenler:**
{causes_section}

🔧 **Tanı Adımları:**
{diagnostic_section}

💡 **Çözüm Adımları:**
{solution_section}

{esde_section}

{warnings_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.ERROR_CODE: """
**Hata Kodu: {error_code}**

📝 **Açıklama:** {error_description}

🔍 **Olası Nedenler:**
{causes_section}

🔧 **Tanı Adımları:**
{diagnostic_section}

💡 **Çözüm:**
{solution_section}

{esde_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.SPECIFICATION: """
**{product_model} - Teknik Özellikler**

📊 **Özellikler:**
{specs_table}

📐 **Boyutlar:**
{dimensions_section}

⚡ **Özellikler:**
{capabilities_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.PROCEDURE: """
**{product_model} - {procedure_title}**

{prerequisites_section}

📋 **Prosedür:**
{procedure_section}

{warnings_section}

✓ **Doğrulama:**
{verification_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.CAPABILITY_QUERY: """
**{product_model} - {capability_name}**

{capability_answer}

{details_section}

{limitations_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.ACCESSORY_QUERY: """
**{product_model} - Aksesuar Bilgileri**

🔌 **Uyumlu Aksesuarlar ({accessory_type}):**
{accessories_list}

{incompatible_section}

{recommendations_section}

📄 **Kaynak:** {sources}
""",
        
        IntentType.GENERAL: """
**{product_model}**

{content}

📄 **Kaynak:** {sources}
""",
    }
    
    # English templates
    TEMPLATES_EN = {
        IntentType.CONFIGURATION: """
**{product_model} - {task_title}**

{prerequisites_section}

📋 **Steps:**
{procedure_section}

{parameter_section}

{warnings_section}

{verification_section}

📄 **Source:** {sources}
""",
        
        IntentType.COMPATIBILITY: """
**{product_model} Compatibility Information**

✅ **Compatible Controllers:**
{controllers_table}

⚙️ **Firmware Requirements:**
{firmware_section}

🔌 **Compatible Accessories:**
{accessories_section}

{incompatible_section}

{recommendations_section}

📄 **Source:** {sources}
""",
        
        IntentType.TROUBLESHOOT: """
**{product_model} - {problem_title}**

🔍 **Possible Causes:**
{causes_section}

🔧 **Diagnostic Steps:**
{diagnostic_section}

💡 **Solution Steps:**
{solution_section}

{esde_section}

{warnings_section}

📄 **Source:** {sources}
""",
        
        IntentType.ERROR_CODE: """
**Error Code: {error_code}**

📝 **Description:** {error_description}

🔍 **Possible Causes:**
{causes_section}

🔧 **Diagnostic Steps:**
{diagnostic_section}

💡 **Solution:**
{solution_section}

{esde_section}

📄 **Source:** {sources}
""",
        
        IntentType.SPECIFICATION: """
**{product_model} - Technical Specifications**

📊 **Specifications:**
{specs_table}

📐 **Dimensions:**
{dimensions_section}

⚡ **Features:**
{capabilities_section}

📄 **Source:** {sources}
""",
        
        IntentType.PROCEDURE: """
**{product_model} - {procedure_title}**

{prerequisites_section}

📋 **Procedure:**
{procedure_section}

{warnings_section}

✓ **Verification:**
{verification_section}

📄 **Source:** {sources}
""",
        
        IntentType.CAPABILITY_QUERY: """
**{product_model} - {capability_name}**

{capability_answer}

{details_section}

{limitations_section}

📄 **Source:** {sources}
""",
        
        IntentType.ACCESSORY_QUERY: """
**{product_model} - Accessory Information**

🔌 **Compatible Accessories ({accessory_type}):**
{accessories_list}

{incompatible_section}

{recommendations_section}

📄 **Source:** {sources}
""",
        
        IntentType.GENERAL: """
**{product_model}**

{content}

📄 **Source:** {sources}
""",
    }
    
    # Language-specific labels
    LABELS = {
        Language.TURKISH: {
            "prerequisites": "Gereksinimler",
            "steps": "Adımlar",
            "procedure": "Prosedür",
            "parameters": "Parametreler",
            "warnings": "Uyarılar",
            "verification": "Doğrulama",
            "source": "Kaynak",
            "compatible_controllers": "Uyumlu Controller'lar",
            "firmware_requirements": "Firmware Gereksinimleri",
            "compatible_accessories": "Uyumlu Aksesuarlar",
            "incompatible": "Uyumsuz",
            "recommendations": "Öneriler",
            "possible_causes": "Olası Nedenler",
            "diagnostic_steps": "Tanı Adımları",
            "solution_steps": "Çözüm Adımları",
            "error_code": "Hata Kodu",
            "description": "Açıklama",
            "solution": "Çözüm",
            "specifications": "Teknik Özellikler",
            "dimensions": "Boyutlar",
            "features": "Özellikler",
            "accessory_info": "Aksesuar Bilgileri",
            "not_found": "Üzgünüm, bu sorgu için yeterli bilgi bulamadım.",
            "suggestions": "Öneriler",
            "greeting": "Merhaba! Ben Desoutter Teknik Asistanı.",
            "greeting_help": "Size nasıl yardımcı olabilirim?",
            "off_topic": "Üzgünüm, bu konu Desoutter ürünleriyle ilgili değil.",
            "off_topic_help": "Desoutter endüstriyel araçları hakkında sorularınızı yanıtlayabilirim.",
            "step": "Adım",
            "min_version": "Min Versiyon",
            "recommended": "Önerilen",
            "tool_min": "Tool",
            "controller_min": "Controller",
            "docks": "Doklar",
            "batteries": "Bataryalar",
            "cables": "Kablolar",
        },
        Language.ENGLISH: {
            "prerequisites": "Prerequisites",
            "steps": "Steps",
            "procedure": "Procedure",
            "parameters": "Parameters",
            "warnings": "Warnings",
            "verification": "Verification",
            "source": "Source",
            "compatible_controllers": "Compatible Controllers",
            "firmware_requirements": "Firmware Requirements",
            "compatible_accessories": "Compatible Accessories",
            "incompatible": "Incompatible",
            "recommendations": "Recommendations",
            "possible_causes": "Possible Causes",
            "diagnostic_steps": "Diagnostic Steps",
            "solution_steps": "Solution Steps",
            "error_code": "Error Code",
            "description": "Description",
            "solution": "Solution",
            "specifications": "Technical Specifications",
            "dimensions": "Dimensions",
            "features": "Features",
            "accessory_info": "Accessory Information",
            "not_found": "Sorry, I couldn't find enough information for this query.",
            "suggestions": "Suggestions",
            "greeting": "Hello! I'm the Desoutter Technical Assistant.",
            "greeting_help": "How can I help you?",
            "off_topic": "Sorry, this topic is not related to Desoutter products.",
            "off_topic_help": "I can answer questions about Desoutter industrial tools.",
            "step": "Step",
            "min_version": "Min Version",
            "recommended": "Recommended",
            "tool_min": "Tool",
            "controller_min": "Controller",
            "docks": "Docks",
            "batteries": "Batteries",
            "cables": "Cables",
        }
    }
    
    # Response templates (backward compatibility - uses Turkish)
    TEMPLATES = TEMPLATES_TR
    
    # Warning icons
    WARNING_ICONS = {
        ValidationStatus.BLOCK: "🚫",
        ValidationStatus.WARN: "⚠️",
        ValidationStatus.ALLOW: "✅",
        ValidationStatus.UNKNOWN: "❓",
    }
    
    def __init__(self, language: Language = Language.TURKISH):
        """
        Initialize Response Formatter
        
        Args:
            language: Output language (default: Turkish)
        """
        self.language = language
        self.templates = self.TEMPLATES_TR if language == Language.TURKISH else self.TEMPLATES_EN
        self.labels = self.LABELS[language]
    
    def set_language(self, language: Language):
        """Change output language"""
        self.language = language
        self.templates = self.TEMPLATES_TR if language == Language.TURKISH else self.TEMPLATES_EN
        self.labels = self.LABELS[language]
    
    def format(
        self,
        intent_result: IntentResult,
        extraction_result: ExtractionResult,
        validation_result: ValidationResult,
        language: Optional[Language] = None
    ) -> FormattedResponse:
        """
        Format extraction result into response.
        
        Args:
            intent_result: Result from Stage 1
            extraction_result: Result from Stage 3
            validation_result: Result from Stage 4
            language: Override language for this response (optional)
            
        Returns:
            FormattedResponse with formatted content
        """
        # Use override language or default
        use_language = language or self.language
        templates = self.TEMPLATES_TR if use_language == Language.TURKISH else self.TEMPLATES_EN
        labels = self.LABELS[use_language]
        intent = intent_result.primary_intent
        
        logger.info(f"Formatting response for intent: {intent.value}")
        
        # Select formatter based on intent
        if intent == IntentType.CONFIGURATION:
            content = self._format_configuration(intent_result, extraction_result, validation_result)
        elif intent == IntentType.COMPATIBILITY:
            content = self._format_compatibility(intent_result, extraction_result, validation_result)
        elif intent in [IntentType.TROUBLESHOOT, IntentType.ERROR_CODE]:
            content = self._format_troubleshoot(intent_result, extraction_result, validation_result)
        elif intent == IntentType.SPECIFICATION:
            content = self._format_specification(intent_result, extraction_result, validation_result)
        elif intent == IntentType.PROCEDURE:
            content = self._format_procedure(intent_result, extraction_result, validation_result)
        elif intent == IntentType.CAPABILITY_QUERY:
            content = self._format_capability(intent_result, extraction_result, validation_result)
        elif intent == IntentType.ACCESSORY_QUERY:
            content = self._format_accessory(intent_result, extraction_result, validation_result)
        else:
            content = self._format_general(intent_result, extraction_result, validation_result)
        
        # Collect warnings from validation
        warnings = self._collect_warnings(validation_result)
        
        # Calculate final confidence
        confidence = extraction_result.confidence + validation_result.confidence_adjustment
        confidence = max(0.0, min(1.0, confidence))
        
        return FormattedResponse(
            content=content.strip(),
            intent=intent,
            product_model=extraction_result.product_model,
            confidence=confidence,
            sources=extraction_result.sources,
            warnings=warnings,
            validation_status=validation_result.status,
            metadata={
                "primary_intent": intent.value,
                "secondary_intents": [s.value for s in intent_result.secondary_intents],
                "entities": {
                    "product_model": intent_result.entities.product_model,
                    "error_code": intent_result.entities.error_code,
                    "parameter_type": intent_result.entities.parameter_type,
                    "target_value": intent_result.entities.target_value,
                }
            }
        )
    
    def _format_configuration(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format configuration response"""
        
        product = extraction.product_model or "Ürün"
        task = self._extract_task_title(intent_result.raw_query)
        
        # Prerequisites section
        prereq_section = ""
        if extraction.prerequisites:
            prereq_parts = ["✅ **Gereksinimler:**"]
            if extraction.prerequisites.controller:
                ctrl = extraction.prerequisites.controller
                ctrl_text = f"- Controller: {ctrl.get('model', 'N/A')}"
                if ctrl.get("min_version"):
                    ctrl_text += f" v{ctrl['min_version']}+"
                prereq_parts.append(ctrl_text)
            if extraction.prerequisites.firmware:
                fw = extraction.prerequisites.firmware
                if fw.get("tool_min"):
                    prereq_parts.append(f"- Tool Firmware: v{fw['tool_min']}+")
                if fw.get("controller_min"):
                    prereq_parts.append(f"- Controller Firmware: v{fw['controller_min']}+")
            if extraction.prerequisites.accessories:
                prereq_parts.append(f"- Aksesuarlar: {', '.join(extraction.prerequisites.accessories)}")
            prereq_section = "\n".join(prereq_parts)
        
        # Procedure section
        procedure_section = self._format_steps(extraction.procedure)
        
        # Parameter ranges section
        param_section = ""
        if extraction.parameter_ranges:
            param_parts = ["⚙️ **Parametre Aralıkları:**"]
            for param in extraction.parameter_ranges:
                min_val = param.min_value if param.min_value is not None else "?"
                max_val = param.max_value if param.max_value is not None else "?"
                unit = param.unit or ""
                param_parts.append(f"- {param.parameter}: {min_val}-{max_val} {unit}")
            param_section = "\n".join(param_parts)
        
        # Warnings section
        warnings_section = self._format_warnings(extraction.warnings, validation)
        
        # Verification section
        verification_section = ""
        if extraction.verification_steps:
            verif_parts = ["✓ **Doğrulama:**"]
            for step in extraction.verification_steps:
                verif_parts.append(f"- {step}")
            verification_section = "\n".join(verif_parts)
        
        # Sources
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.CONFIGURATION].format(
            product_model=product,
            task_title=task,
            prerequisites_section=prereq_section,
            procedure_section=procedure_section,
            parameter_section=param_section,
            warnings_section=warnings_section,
            verification_section=verification_section,
            sources=sources
        )
    
    def _format_compatibility(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format compatibility response"""
        
        product = extraction.product_model or "Ürün"
        
        # Controllers table
        controllers_table = ""
        if extraction.compatibility and extraction.compatibility.compatible_controllers:
            table_rows = ["| Controller | Min Versiyon | Önerilen |", "|------------|--------------|----------|"]
            for ctrl in extraction.compatibility.compatible_controllers:
                model = ctrl.get("model", "N/A")
                version = ctrl.get("min_version", "N/A")
                recommended = "✅" if ctrl.get("recommended") else "-"
                table_rows.append(f"| {model} | v{version} | {recommended} |")
            controllers_table = "\n".join(table_rows)
        
        # Firmware section
        firmware_section = ""
        if extraction.compatibility and extraction.compatibility.firmware_requirements:
            fw = extraction.compatibility.firmware_requirements
            fw_parts = []
            if fw.get("tool_min"):
                fw_parts.append(f"- Tool: v{fw['tool_min']}+")
            if fw.get("controller_min"):
                fw_parts.append(f"- Controller: v{fw['controller_min']}+")
            firmware_section = "\n".join(fw_parts)
        
        # Accessories section
        accessories_section = ""
        if extraction.compatibility and extraction.compatibility.compatible_accessories:
            acc = extraction.compatibility.compatible_accessories
            acc_parts = []
            if acc.get("docks"):
                acc_parts.append(f"- **Doklar:** {', '.join(acc['docks'])}")
            if acc.get("batteries"):
                acc_parts.append(f"- **Bataryalar:** {', '.join(acc['batteries'])}")
            if acc.get("cables"):
                acc_parts.append(f"- **Kablolar:** {', '.join(acc['cables'])}")
            accessories_section = "\n".join(acc_parts)
        
        # Incompatible section
        incompatible_section = ""
        if extraction.compatibility and extraction.compatibility.incompatible_items:
            incompatible_section = "❌ **Uyumsuz:**\n" + "\n".join(
                f"- {item}" for item in extraction.compatibility.incompatible_items
            )
        
        # Recommendations
        recommendations_section = ""
        if extraction.compatibility and extraction.compatibility.recommendations:
            recommendations_section = "📌 **Öneriler:**\n" + "\n".join(
                f"- {rec}" for rec in extraction.compatibility.recommendations
            )
        
        # Add validation-based recommendations
        for issue in validation.issues:
            if issue.status == ValidationStatus.ALLOW and "controller" in issue.field:
                recommendations_section += f"\n✅ {issue.message}"
        
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.COMPATIBILITY].format(
            product_model=product,
            controllers_table=controllers_table,
            firmware_section=firmware_section,
            accessories_section=accessories_section,
            incompatible_section=incompatible_section,
            recommendations_section=recommendations_section,
            sources=sources
        )
    
    def _format_troubleshoot(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format troubleshooting response"""
        
        product = extraction.product_model or "Ürün"
        
        if extraction.troubleshoot:
            ts = extraction.troubleshoot
            problem_title = ts.problem_description[:50] + "..." if len(ts.problem_description) > 50 else ts.problem_description
            
            # Causes
            causes_section = "\n".join(f"{i+1}. {cause}" for i, cause in enumerate(ts.possible_causes))
            
            # Diagnostic steps
            diagnostic_section = "\n".join(f"{i+1}. {step}" for i, step in enumerate(ts.diagnostic_steps))
            
            # Solutions
            solution_section = self._format_steps(ts.solutions)
            
            # ESDE section
            esde_section = ""
            if ts.esde_reference:
                esde_section = f"""
⚠️ **ESDE Servisi:**
**{ts.esde_reference}**: {ts.esde_description or 'Bilinen üretim hatası'}
Bu bilinen bir üretim hatasıdır. Servis müdahalesi gereklidir.
"""
                if ts.affected_models:
                    esde_section += f"\n**Etkilenen Modeller:** {', '.join(ts.affected_models)}"
            
        else:
            problem_title = "Sorun"
            causes_section = "Bilgi bulunamadı"
            diagnostic_section = "Bilgi bulunamadı"
            solution_section = "Bilgi bulunamadı"
            esde_section = ""
        
        warnings_section = self._format_warnings(extraction.warnings, validation)
        sources = self._format_sources(extraction.sources)
        
        # Use ERROR_CODE template if error code is present
        if intent_result.entities.error_code:
            return self.TEMPLATES[IntentType.ERROR_CODE].format(
                error_code=intent_result.entities.error_code,
                error_description=problem_title,
                causes_section=causes_section,
                diagnostic_section=diagnostic_section,
                solution_section=solution_section,
                esde_section=esde_section,
                sources=sources
            )
        
        return self.TEMPLATES[IntentType.TROUBLESHOOT].format(
            product_model=product,
            problem_title=problem_title,
            causes_section=causes_section,
            diagnostic_section=diagnostic_section,
            solution_section=solution_section,
            esde_section=esde_section,
            warnings_section=warnings_section,
            sources=sources
        )
    
    def _format_specification(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format specification response"""
        
        product = extraction.product_model or "Ürün"
        specs = extraction.specifications
        
        # Specs table
        specs_table = ""
        if specs:
            table_rows = ["| Özellik | Değer |", "|---------|-------|"]
            for key, value in specs.items():
                if isinstance(value, dict):
                    if "min" in value and "max" in value:
                        val_str = f"{value.get('min', '?')}-{value.get('max', '?')} {value.get('unit', '')}"
                    else:
                        val_str = str(value)
                elif isinstance(value, list):
                    val_str = ", ".join(str(v) for v in value)
                else:
                    val_str = str(value)
                table_rows.append(f"| {key} | {val_str} |")
            specs_table = "\n".join(table_rows)
        
        dimensions_section = "Boyut bilgisi mevcut değil"
        capabilities_section = "Özellik bilgisi mevcut değil"
        
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.SPECIFICATION].format(
            product_model=product,
            specs_table=specs_table,
            dimensions_section=dimensions_section,
            capabilities_section=capabilities_section,
            sources=sources
        )
    
    def _format_procedure(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format procedure response"""
        
        product = extraction.product_model or "Ürün"
        task = self._extract_task_title(intent_result.raw_query)
        
        # Prerequisites
        prereq_section = ""
        if extraction.prerequisites:
            prereq_parts = ["✅ **Gereksinimler:**"]
            if extraction.prerequisites.accessories:
                prereq_parts.append(f"- Araçlar: {', '.join(extraction.prerequisites.accessories)}")
            if extraction.prerequisites.warnings:
                for warn in extraction.prerequisites.warnings:
                    prereq_parts.append(f"- ⚠️ {warn}")
            prereq_section = "\n".join(prereq_parts)
        
        # Procedure
        procedure_section = self._format_steps(extraction.procedure)
        
        # Warnings
        warnings_section = self._format_warnings(extraction.warnings, validation)
        
        # Verification
        verification_section = ""
        if extraction.verification_steps:
            verification_section = "\n".join(f"- {step}" for step in extraction.verification_steps)
        
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.PROCEDURE].format(
            product_model=product,
            procedure_title=task,
            prerequisites_section=prereq_section,
            procedure_section=procedure_section,
            warnings_section=warnings_section,
            verification_section=verification_section,
            sources=sources
        )
    
    def _format_capability(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format capability query response"""
        
        product = extraction.product_model or "Ürün"
        capability = intent_result.raw_query
        
        # Simple yes/no answer based on extraction
        raw = extraction.raw_extraction or {}
        supported = raw.get("supported", False)
        details = raw.get("details", "Detaylı bilgi bulunamadı")
        limitations = raw.get("limitations", [])
        
        if supported:
            answer = f"✅ Evet, {product} bu özelliği destekliyor."
        else:
            answer = f"❌ Hayır, {product} bu özelliği desteklemiyor."
        
        details_section = f"📝 **Detay:** {details}" if details else ""
        
        limitations_section = ""
        if limitations:
            limitations_section = "⚠️ **Kısıtlamalar:**\n" + "\n".join(f"- {lim}" for lim in limitations)
        
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.CAPABILITY_QUERY].format(
            product_model=product,
            capability_name=capability[:30],
            capability_answer=answer,
            details_section=details_section,
            limitations_section=limitations_section,
            sources=sources
        )
    
    def _format_accessory(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format accessory query response"""
        
        product = extraction.product_model or "Ürün"
        accessory_type = intent_result.entities.accessory_type or "Aksesuar"
        
        raw = extraction.raw_extraction or {}
        accessories = raw.get("compatible_accessories", [])
        incompatible = raw.get("incompatible", [])
        recommendations = raw.get("recommendations", [])
        
        accessories_list = ""
        if accessories:
            for acc in accessories:
                if isinstance(acc, dict):
                    name = acc.get("name", "N/A")
                    model = acc.get("model", "")
                    notes = acc.get("notes", "")
                    accessories_list += f"- **{name}** {model} {notes}\n"
                else:
                    accessories_list += f"- {acc}\n"
        else:
            accessories_list = "Bilgi bulunamadı"
        
        incompatible_section = ""
        if incompatible:
            incompatible_section = "❌ **Uyumsuz:**\n" + "\n".join(f"- {item}" for item in incompatible)
        
        recommendations_section = ""
        if recommendations:
            recommendations_section = "📌 **Öneriler:**\n" + "\n".join(f"- {rec}" for rec in recommendations)
        
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.ACCESSORY_QUERY].format(
            product_model=product,
            accessory_type=accessory_type,
            accessories_list=accessories_list,
            incompatible_section=incompatible_section,
            recommendations_section=recommendations_section,
            sources=sources
        )
    
    def _format_general(
        self,
        intent_result: IntentResult,
        extraction: ExtractionResult,
        validation: ValidationResult
    ) -> str:
        """Format general response"""
        
        product = extraction.product_model or "Genel"
        
        raw = extraction.raw_extraction or {}
        summary = raw.get("summary", "")
        key_points = raw.get("key_points", [])
        
        content_parts = []
        if summary:
            content_parts.append(summary)
        if key_points:
            content_parts.append("\n**Önemli Noktalar:**")
            for point in key_points:
                content_parts.append(f"- {point}")
        
        content = "\n".join(content_parts) if content_parts else "Detaylı bilgi bulunamadı."
        
        sources = self._format_sources(extraction.sources)
        
        return self.TEMPLATES[IntentType.GENERAL].format(
            product_model=product,
            content=content,
            sources=sources
        )
    
    def _format_steps(self, steps: List[ProcedureStep]) -> str:
        """Format procedure steps"""
        if not steps:
            return "Adım bilgisi bulunamadı"
        
        formatted = []
        for step in steps:
            step_text = f"{step.step_number}. {step.action}"
            if step.details:
                step_text += f"\n   {step.details}"
            if step.warning:
                step_text += f"\n   ⚠️ {step.warning}"
            if step.expected_result:
                step_text += f"\n   → {step.expected_result}"
            formatted.append(step_text)
        
        return "\n".join(formatted)
    
    def _format_warnings(
        self,
        warnings: List[str],
        validation: ValidationResult
    ) -> str:
        """Format warnings from extraction and validation"""
        all_warnings = list(warnings)
        
        # Add validation warnings
        for issue in validation.issues:
            if issue.status in [ValidationStatus.WARN, ValidationStatus.BLOCK]:
                icon = self.WARNING_ICONS.get(issue.status, "⚠️")
                warning_text = f"{icon} {issue.message}"
                if issue.suggestion:
                    warning_text += f" → {issue.suggestion}"
                all_warnings.append(warning_text)
        
        if not all_warnings:
            return ""
        
        return "⚠️ **Uyarılar:**\n" + "\n".join(f"- {w}" for w in all_warnings)
    
    def _format_sources(self, sources: List[str]) -> str:
        """Format source citations"""
        if not sources:
            return "Belirtilmemiş"
        
        # Limit to 3 sources
        limited = sources[:3]
        return ", ".join(limited)
    
    def _extract_task_title(self, query: str) -> str:
        """Extract task title from query"""
        # Remove common question words
        title = query
        for word in ["nasıl", "how to", "ne şekilde", "?", "."]:
            title = title.replace(word, "")
        
        return title.strip().capitalize() or "İşlem"
    
    def _collect_warnings(self, validation: ValidationResult) -> List[str]:
        """Collect warning messages from validation"""
        warnings = []
        for issue in validation.issues:
            if issue.status in [ValidationStatus.WARN, ValidationStatus.BLOCK]:
                warnings.append(issue.message)
        return warnings
    
    def format_no_result(self, intent_result: IntentResult) -> FormattedResponse:
        """Format response when no results found"""
        product = intent_result.entities.product_model or "belirtilen ürün"
        
        content = f"""
Üzgünüm, **{product}** için bu sorguyla ilgili yeterli bilgi bulamadım.

**Öneriler:**
- Ürün modelini tam olarak belirtin (örn: EABC-3000)
- Hata kodu varsa ekleyin
- Soruyu daha spesifik hale getirin

Başka bir konuda yardımcı olabilir miyim?
"""
        
        return FormattedResponse(
            content=content.strip(),
            intent=intent_result.primary_intent,
            product_model=intent_result.entities.product_model,
            confidence=0.0,
            sources=[],
            warnings=["Yeterli bilgi bulunamadı"],
            validation_status=ValidationStatus.UNKNOWN,
        )
    
    def format_off_topic(self, intent_result: IntentResult) -> FormattedResponse:
        """Format response for off-topic queries"""
        content = """
Bu soru Desoutter endüstriyel araçları ile ilgili değil gibi görünüyor.

Ben Desoutter teknik destek asistanıyım. Size şu konularda yardımcı olabilirim:
- 🔧 Araç arızaları ve hata kodları
- ⚙️ Konfigürasyon ve parametre ayarları
- 🔌 Uyumluluk soruları (tool-controller-aksesuar)
- 📋 Bakım prosedürleri
- 📊 Teknik özellikler

Desoutter araçlarıyla ilgili bir sorunuz var mı?
"""
        
        return FormattedResponse(
            content=content.strip(),
            intent=IntentType.OFF_TOPIC,
            product_model=None,
            confidence=1.0,
            sources=[],
            warnings=[],
            validation_status=ValidationStatus.ALLOW,
        )
    
    def format_greeting(self, intent_result: IntentResult) -> FormattedResponse:
        """Format response for greetings"""
        content = """
Merhaba! 👋

Ben Desoutter Teknik Destek Asistanı'yım. Size şu konularda yardımcı olabilirim:

- 🔧 **Arıza Teşhisi:** Hata kodları ve sorun giderme
- ⚙️ **Konfigürasyon:** Pset ayarları, parametre yapılandırma
- 🔌 **Uyumluluk:** Tool-controller-aksesuar uyumluluğu
- 📋 **Prosedürler:** Kurulum, bakım, kalibrasyon
- 📊 **Teknik Özellikler:** Ürün spesifikasyonları

Nasıl yardımcı olabilirim?
"""
        
        return FormattedResponse(
            content=content.strip(),
            intent=IntentType.GREETING,
            product_model=None,
            confidence=1.0,
            sources=[],
            warnings=[],
            validation_status=ValidationStatus.ALLOW,
        )
