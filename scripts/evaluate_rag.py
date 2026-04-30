#!/usr/bin/env python3
"""
RAG Evaluation Script — Q&A dataset üzerinde sistem kalitesini ölçer.

Her Q&A çifti için:
1. /api/diagnose endpoint'i çağırır
2. Cevabı expected_answer ile karşılaştırır
3. Keyword overlap, source presence, response quality metriklerini hesaplar

Output: evaluation_results.json + evaluation_summary.json

Usage:
    python scripts/evaluate_rag.py --limit 5
    python scripts/evaluate_rag.py --limit 50 --api-url http://localhost:8000
    python scripts/evaluate_rag.py --input /app/data/qa_dataset.json --output /app/data/eval_results.json
"""
import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_keywords(text: str, min_len: int = 4) -> set:
    """Extract meaningful words from text for overlap comparison."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # strip HTML
    words = re.findall(r'\b[a-z][a-z0-9\-]{%d,}\b' % (min_len - 1), text)
    stopwords = {
        'that', 'this', 'with', 'from', 'have', 'will', 'your', 'been',
        'they', 'their', 'there', 'about', 'which', 'when', 'also', 'some',
        'more', 'than', 'then', 'into', 'other', 'what', 'would', 'could',
        'should', 'these', 'those', 'such', 'each', 'both', 'here', 'after',
        'before', 'where', 'while', 'very', 'just', 'over', 'tool', 'unit',
        'please', 'hello', 'thank', 'regards', 'dear', 'need', 'know',
    }
    return {w for w in words if w not in stopwords}


def keyword_overlap_score(expected: str, actual: str) -> float:
    """Compute keyword overlap between expected and actual answer (0.0-1.0)."""
    exp_kw = extract_keywords(expected)
    if not exp_kw:
        return 0.0
    act_kw = extract_keywords(actual)
    overlap = exp_kw & act_kw
    return len(overlap) / len(exp_kw)


def has_source_citation(response_text: str) -> bool:
    """Check if the response includes a source reference."""
    return bool(re.search(r'source|ticket|bulletin|esde|document|pdf', response_text.lower()))


def is_hallucination_risk(actual: str, expected: str) -> bool:
    """Flag if actual answer contains error codes not in expected."""
    actual_codes = set(re.findall(r'[EW]\d{3,4}', actual, re.IGNORECASE))
    expected_codes = set(re.findall(r'[EW]\d{3,4}', expected, re.IGNORECASE))
    if actual_codes and not (actual_codes & expected_codes):
        return True
    return False


def call_diagnose(api_url: str, question: str, model_name: str, timeout: int = 60) -> dict:
    """Call /api/diagnose endpoint and return result."""
    payload = {
        "part_number": "",
        "fault_description": question,
        "language": "en",
    }
    try:
        resp = requests.post(
            f"{api_url}/diagnose",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout", "answer": ""}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "answer": ""}


def evaluate_single(qa: dict, api_url: str) -> dict:
    """Evaluate a single Q&A pair. Returns evaluation record."""
    question = qa["question"]
    expected = qa["expected_answer"]
    models = qa.get("related_models", [])
    model_name = models[0] if models else ""

    start = time.time()
    result = call_diagnose(api_url, question, model_name)
    elapsed = time.time() - start

    actual = result.get("suggestion", "") or result.get("answer", "") or result.get("response", "") or ""

    overlap = keyword_overlap_score(expected, actual)
    has_answer = len(actual.strip()) > 50
    has_source = has_source_citation(actual)
    hallucination = is_hallucination_risk(actual, expected)
    error = result.get("error")

    if error or not has_answer:
        grade = 0
    elif overlap >= 0.15:
        grade = 2
    elif overlap >= 0.07:
        grade = 1
    else:
        grade = 0

    return {
        "id": qa["id"],
        "question": question[:200],
        "expected_answer": expected[:300],
        "actual_answer": actual[:500],
        "model_name": model_name,
        "language": qa.get("language", "en"),
        "related_models": models,
        "keyword_overlap": round(overlap, 3),
        "has_answer": has_answer,
        "has_source": has_source,
        "hallucination_risk": hallucination,
        "grade": grade,
        "grade_label": ["fail", "partial", "good"][grade],
        "elapsed_sec": round(elapsed, 2),
        "error": error,
    }


def build_summary(results: list) -> dict:
    """Build aggregate summary from evaluation results."""
    total = len(results)
    if total == 0:
        return {}

    overlaps = [r["keyword_overlap"] for r in results]
    errors = [r for r in results if r.get("error")]
    good = [r for r in results if r["grade"] == 2]
    partial = [r for r in results if r["grade"] == 1]
    fail = [r for r in results if r["grade"] == 0]
    hallucinations = [r for r in results if r["hallucination_risk"]]

    avg_overlap = sum(overlaps) / total
    avg_elapsed = sum(r["elapsed_sec"] for r in results) / total

    model_grades = {}
    for r in results:
        m = r["model_name"] or "unknown"
        if m not in model_grades:
            model_grades[m] = {"good": 0, "partial": 0, "fail": 0, "total": 0}
        model_grades[m][r["grade_label"]] += 1
        model_grades[m]["total"] += 1

    hardest = sorted(
        [r for r in results if not r.get("error")],
        key=lambda x: x["keyword_overlap"]
    )[:20]

    return {
        "evaluated_at": datetime.now().isoformat(),
        "total": total,
        "good": len(good),
        "partial": len(partial),
        "fail": len(fail),
        "errors": len(errors),
        "hallucination_risk": len(hallucinations),
        "good_pct": round(len(good) / total * 100, 1),
        "partial_pct": round(len(partial) / total * 100, 1),
        "fail_pct": round(len(fail) / total * 100, 1),
        "avg_keyword_overlap": round(avg_overlap, 3),
        "avg_elapsed_sec": round(avg_elapsed, 2),
        "per_model": dict(
            sorted(model_grades.items(), key=lambda x: -x[1]["total"])[:15]
        ),
        "hardest_questions": [
            {"id": r["id"], "question": r["question"][:100], "overlap": r["keyword_overlap"]}
            for r in hardest
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system on Q&A dataset')
    parser.add_argument('--input', type=str, default='/app/data/qa_dataset.json',
                        help='Q&A dataset JSON file')
    parser.add_argument('--output', type=str, default='/app/data/eval_results.json',
                        help='Output evaluation results JSON')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000',
                        help='API base URL')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of Q&A pairs to evaluate')
    parser.add_argument('--offset', type=int, default=0,
                        help='Start from this index in the dataset (default: 0)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between requests in seconds (default: 0.5)')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.offset:
        dataset = dataset[args.offset:]
    if args.limit:
        dataset = dataset[:args.limit]

    logger.info(f"Evaluating {len(dataset)} Q&A pairs against {args.api_url}")

    results = []
    for i, qa in enumerate(dataset):
        logger.info(f"[{i+1}/{len(dataset)}] Ticket #{qa['id']} — {qa['question'][:60]}...")
        result = evaluate_single(qa, args.api_url)
        results.append(result)
        logger.info(f"  grade={result['grade_label']} overlap={result['keyword_overlap']} elapsed={result['elapsed_sec']}s")
        if args.delay > 0 and i < len(dataset) - 1:
            time.sleep(args.delay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(results)} evaluation results to {output_path}")

    summary = build_summary(results)
    summary_path = output_path.with_name('eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total:    {summary['total']}")
    logger.info(f"Good:     {summary['good']} ({summary['good_pct']}%)")
    logger.info(f"Partial:  {summary['partial']} ({summary['partial_pct']}%)")
    logger.info(f"Fail:     {summary['fail']} ({summary['fail_pct']}%)")
    logger.info(f"Errors:   {summary['errors']}")
    logger.info(f"Avg overlap: {summary['avg_keyword_overlap']}")
    logger.info(f"Avg response time: {summary['avg_elapsed_sec']}s")
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
