"""
El-Harezmi Response Normalizer
================================
Legacy RAGEngine response schema (generate_repair_suggestion() → Dict) ile
ElHarezmi PipelineResult schema'sını birleştiren normalizer.

API katmanı her zaman aynı DiagnoseResponse dict yapısını görür.

Schema Analizi (ADIM 0):
  Legacy alanlar:
    suggestion, confidence (str), confidence_numeric (float),
    product_model, part_number, sources (List[dict]),
    language (str), diagnosis_id, response_time_ms,
    intent (str), intent_confidence (float),
    sufficiency_score, sufficiency_reason, sufficiency_factors,
    sufficiency_recommendation, validation (dict)

  El-Harezmi (FormattedResponse) alanlar:
    content (str), intent (IntentType), product_model, confidence (float),
    sources (List[str]), warnings (List[str]),
    validation_status (ValidationStatus), language (Language),
    timestamp, metadata (dict)
"""

from typing import Dict, Any, Optional
from .pipeline import PipelineResult


def normalize_el_harezmi_result(
    result: PipelineResult,
    original_part_number: Optional[str] = None,
    original_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    PipelineResult → legacy rag_engine.generate_repair_suggestion() ile aynı dict yapısı.

    Args:
        result: ElHarezmiPipeline.process() dönüş değeri
        original_part_number: İstek'ten gelen orijinal part_number
        original_language: İstek'ten gelen orijinal language kodu ("en"/"tr")

    Returns:
        DiagnoseResponse modeline uyumlu dict
    """
    resp = result.response
    intent_result = result.intent_result
    metrics = result.metrics

    # ---------------------------------------------------------------
    # 1. suggestion = FormattedResponse.content (markdown metin)
    #    Legacy: "suggestion" string alanı
    # ---------------------------------------------------------------
    suggestion = resp.content if resp.content else "Yanıt üretilemedi."

    # ---------------------------------------------------------------
    # 2. confidence: float → "high"/"medium"/"low" string dönüşümü
    #    Legacy: "high" | "medium" | "low" | "insufficient_context"
    #    El-Harezmi: 0.0-1.0 float
    #    Eşik: 0.7+ = high, 0.5+ = medium, <0.5 = low
    # ---------------------------------------------------------------
    confidence_float = float(resp.confidence) if resp.confidence is not None else 0.0

    if confidence_float >= 0.7:
        confidence_str = "high"
    elif confidence_float >= 0.5:
        confidence_str = "medium"
    else:
        # El-Harezmi pipeline başarısız olduysa insufficient_context
        if not result.success:
            confidence_str = "insufficient_context"
        else:
            confidence_str = "low"

    # ---------------------------------------------------------------
    # 3. product_model: FormattedResponse.product_model
    #    Fallback: intent_result.entities üzerinden veya part_number
    # ---------------------------------------------------------------
    product_model = (
        resp.product_model
        or (intent_result.entities.product_model if intent_result.entities else None)
        or original_part_number
        or "Unknown"
    )

    # ---------------------------------------------------------------
    # 4. part_number: El-Harezmi'de yok → original_part_number kullan
    #    Default "": API'ye gelen part_number'ı geçir
    # ---------------------------------------------------------------
    part_number = original_part_number or product_model

    # ---------------------------------------------------------------
    # 5. sources: List[str] → List[dict] dönüşümü
    #    Legacy: [{"source": str, "page": int, "section": str,
    #              "similarity": str, "excerpt": str}]
    #    El-Harezmi: ["source_name_1", "source_name_2", ...]
    #    Retrieval chunks'lardan similarity/excerpt eklenebilirse ekle
    # ---------------------------------------------------------------
    sources_list = _build_sources_list(result)

    # ---------------------------------------------------------------
    # 6. language: Language enum → "en"/"tr" string
    #    Fallback: original_language → "en"
    # ---------------------------------------------------------------
    try:
        language_str = resp.language.value  # Language.TURKISH → "tr"
    except Exception:
        language_str = original_language or "tr"

    # ---------------------------------------------------------------
    # 7. diagnosis_id: El-Harezmi bu alanı üretmiyor
    #    None: feedback sistemiyle henüz entegre değil
    # ---------------------------------------------------------------
    diagnosis_id = None  # TODO: Feedback entegrasyonu eklenince doldurulacak

    # ---------------------------------------------------------------
    # 8. response_time_ms: metrics.total_time_ms float → int
    # ---------------------------------------------------------------
    response_time_ms = int(metrics.total_time_ms)

    # ---------------------------------------------------------------
    # 9. intent: IntentType enum → value string
    #    Legacy: "general", "troubleshoot", "error_code" vb.
    # ---------------------------------------------------------------
    intent_str = intent_result.primary_intent.value if intent_result.primary_intent else "general"

    # ---------------------------------------------------------------
    # 10. intent_confidence: intent_result.confidence float
    #     Default 0.0 eğer classification yapılmadıysa
    # ---------------------------------------------------------------
    intent_confidence = float(intent_result.confidence) if intent_result.confidence else 0.0

    # ---------------------------------------------------------------
    # 11. validation: El-Harezmi'nin KG validation sonucu
    #     Legacy format: {"issues": [...], "severity": "..."}
    #     None yerine structured dict döndür
    # ---------------------------------------------------------------
    validation = _build_validation_dict(result)

    # ---------------------------------------------------------------
    # 12. sufficiency_*: El-Harezmi'de bu konsept yok
    #     None: API'ye opsiyonel field olarak geçer, hata olmaz
    # ---------------------------------------------------------------
    # (Legacy opsiyonel alanlar — None default yeterli)

    # ---------------------------------------------------------------
    # Temel yapıyı oluştur (DiagnoseResponse ile uyumlu)
    # ---------------------------------------------------------------
    normalized: Dict[str, Any] = {
        "suggestion": suggestion,
        "confidence": confidence_str,
        "confidence_numeric": confidence_float,
        "product_model": product_model,
        "part_number": part_number,
        "sources": sources_list,
        "language": language_str,
        "diagnosis_id": diagnosis_id,
        "response_time_ms": response_time_ms,
        "intent": intent_str,
        "intent_confidence": intent_confidence,
        "validation": validation,
        # Opsiyonel: El-Harezmi içsel metadata
        "el_harezmi_metadata": {
            "stage1_ms": metrics.stage1_time_ms,
            "stage2_ms": metrics.stage2_time_ms,
            "stage3_ms": metrics.stage3_time_ms,
            "stage4_ms": metrics.stage4_time_ms,
            "stage5_ms": metrics.stage5_time_ms,
            "chunks_retrieved": metrics.chunks_retrieved,
            "validation_status": metrics.validation_status,
            "secondary_intents": (
                [s.value for s in intent_result.secondary_intents]
                if intent_result.secondary_intents else []
            ),
            "warnings": resp.warnings,
            "success": result.success,
            "error": result.error,
        }
    }

    return normalized


def _build_sources_list(result: PipelineResult) -> list:
    """
    El-Harezmi retrieval chunks'larından legacy sources dict listesi oluştur.

    El-Harezmi'nin iki kaynak yolu:
    1. result.retrieval_result.chunks → chunk metadata'sından detaylar
    2. result.response.sources → sadece isim listesi (fallback)
    """
    sources = []
    seen = set()

    # Önce retrieval_result'tan detaylı chunk bilgisi dene
    if result.retrieval_result and result.retrieval_result.chunks:
        for chunk in result.retrieval_result.chunks[:5]:  # Max 5 kaynak
            source_name = (
                chunk.metadata.get("source", "Unknown") 
                if chunk.metadata else "Unknown"
            )

            if source_name in seen:
                continue
            seen.add(source_name)

            # similarity: chunk.score (0.0-1.0 float) → "0.85" string
            similarity = f"{chunk.score:.2f}" if hasattr(chunk, "score") and chunk.score else "0.00"

            # excerpt: chunk.text'in ilk 200 karakteri
            text = getattr(chunk, "text", "") or ""
            excerpt = text[:200] + "..." if len(text) > 200 else text

            sources.append({
                "source": source_name,
                "page": chunk.metadata.get("page_number") if chunk.metadata else None,
                "section": chunk.metadata.get("section_hierarchy", "") if chunk.metadata else "",
                "similarity": similarity,
                "excerpt": excerpt,
            })

    # Eğer retrieval yoksa response.sources string listesini kullan
    elif result.response.sources:
        for source_name in result.response.sources[:5]:
            if source_name in seen:
                continue
            seen.add(source_name)

            sources.append({
                "source": source_name,
                "page": None,      # El-Harezmi string listesinde sayfa yok
                "section": "",     # Section bilgisi yok
                "similarity": "0.00",  # Score bilgisi yok
                "excerpt": "",     # Excerpt yok
            })

    return sources


def _build_validation_dict(result: PipelineResult) -> dict:
    """
    El-Harezmi Stage 4 KG validation sonucunu legacy validation dict'e çevir.

    Legacy format (ResponseValidator):
        {"issues": [...], "severity": "low"/"medium"/"high", ...}

    El-Harezmi format (ValidationResult):
        status: ValidationStatus, issues: [ValidationIssue], ...
    """
    if not result.validation_result:
        return None  # type: ignore  # Opsiyonel alan, None kabul edilir

    vr = result.validation_result

    issues = []
    for issue in (vr.issues or []):
        issues.append({
            "type": issue.field,
            "description": issue.message,
            "severity": issue.status.value,
            "suggestion": issue.suggestion,
        })

    # Severity: BLOCK > WARN > ALLOW
    from .stage4_kg_validation import ValidationStatus
    if vr.status == ValidationStatus.BLOCK:
        severity = "high"
    elif vr.status == ValidationStatus.WARN:
        severity = "medium"
    else:
        severity = "low"

    return {
        "status": vr.status.value,
        "severity": severity,
        "issues": issues,
        "confidence_adjustment": vr.confidence_adjustment,
    }
