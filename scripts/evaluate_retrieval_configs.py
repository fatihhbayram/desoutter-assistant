#!/usr/bin/env python3
"""
Thesis Experiment: 4 Retrieval Configuration Comparison

Konfigürasyonlar:
  A — Semantic-only    : Dense vector search (Qdrant), BM25 devre dışı
  B — BM25-only        : Sparse keyword search, semantic devre dışı
  C — TF-IDF           : sklearn TfidfVectorizer + cosine similarity
  D — Hybrid (current) : BM25 + Semantic + RRF (production sistemi)

Metrik: Keyword Overlap@5 — top-5 retrieved chunk'ların expected_answer
        ile keyword örtüşmesi. LLM generation yok — retrieval kalitesi izole ölçülüyor.

Usage:
    docker exec desoutter-api python scripts/evaluate_retrieval_configs.py
    docker exec desoutter-api python scripts/evaluate_retrieval_configs.py --limit 100
    docker exec desoutter-api python scripts/evaluate_retrieval_configs.py --output /app/data/retrieval_comparison.json
"""
import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.vectordb.qdrant_client import QdrantDBClient
from src.documents.embeddings import EmbeddingsGenerator
from src.llm.hybrid_search import HybridSearcher, BM25Index, SearchResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION = 'desoutter_docs_v2'
QA_DATASET_PATH = '/app/data/qa_dataset.json'

STOPWORDS = {
    'that', 'this', 'with', 'from', 'have', 'will', 'your', 'been',
    'they', 'their', 'there', 'about', 'which', 'when', 'also', 'some',
    'more', 'than', 'then', 'into', 'other', 'what', 'would', 'could',
    'should', 'these', 'those', 'such', 'each', 'both', 'here', 'after',
    'before', 'where', 'while', 'very', 'just', 'over', 'tool', 'unit',
    'please', 'hello', 'thank', 'regards', 'dear', 'need', 'know',
}


def extract_keywords(text: str, min_len: int = 4) -> set:
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    words = re.findall(r'\b[a-z][a-z0-9\-]{%d,}\b' % (min_len - 1), text)
    return {w for w in words if w not in STOPWORDS}


def keyword_overlap(expected: str, chunks: list) -> float:
    """Keyword overlap between expected answer and retrieved chunk texts."""
    exp_kw = extract_keywords(expected)
    if not exp_kw:
        return 0.0
    retrieved_text = ' '.join(c.get('content', '') for c in chunks)
    ret_kw = extract_keywords(retrieved_text)
    return len(exp_kw & ret_kw) / len(exp_kw)


def grade(score: float) -> str:
    if score >= 0.15:
        return 'good'
    elif score >= 0.07:
        return 'partial'
    return 'fail'


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval functions (one per config)
# ──────────────────────────────────────────────────────────────────────────────

class SemanticRetriever:
    """Config A: Dense-only search via Qdrant."""

    def __init__(self, qdrant: QdrantDBClient, embedder: EmbeddingsGenerator):
        self.qdrant = qdrant
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list:
        embedding = self.embedder.generate_embeddings([query])[0]
        results = self.qdrant.client.search(
            collection_name=self.qdrant.collection_name,
            query_vector=('dense', embedding),
            limit=top_k,
        )
        return [
            {'id': str(p.id), 'content': (p.payload or {}).get('text', ''), 'score': p.score}
            for p in results
        ]


class BM25Retriever:
    """Config B: BM25 sparse-only search."""

    def __init__(self, bm25_index: BM25Index, doc_metadata: dict):
        self.bm25 = bm25_index
        self.meta = doc_metadata

    def search(self, query: str, top_k: int = 5) -> list:
        results = self.bm25.search(query, top_k=top_k)
        return [
            {'id': r.id, 'content': r.content, 'score': r.bm25_score}
            for r in results
        ]


class TFIDFRetriever:
    """Config C: TF-IDF + cosine similarity."""

    def __init__(self, all_docs: list):
        logger.info("Building TF-IDF index...")
        self.docs = all_docs
        self.ids = [d['id'] for d in all_docs]
        texts = [d['content'] for d in all_docs]
        self.vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"  TF-IDF matrix: {self.matrix.shape}")

    def search(self, query: str, top_k: int = 5) -> list:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {'id': self.ids[i], 'content': self.docs[i]['content'], 'score': float(scores[i])}
            for i in top_idx
        ]


class HybridRetriever:
    """Config D: BM25 + Semantic + RRF (mevcut üretim sistemi)."""

    def __init__(self, searcher: HybridSearcher):
        self.searcher = searcher

    def search(self, query: str, top_k: int = 5) -> list:
        results = self.searcher.search(
            query=query,
            top_k=top_k,
            expand_query=False,
            use_hybrid=True,
            min_similarity=0.0,
        )
        return [
            {'id': r.id, 'content': r.content, 'score': r.score}
            for r in results
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_config(retriever, qa_pairs: list, config_name: str, top_k: int = 5) -> dict:
    logger.info(f"\n{'='*60}")
    logger.info(f"Config {config_name} — {len(qa_pairs)} sorgu")
    logger.info('='*60)

    records = []
    grades = defaultdict(int)
    total_overlap = 0.0
    errors = 0

    for i, qa in enumerate(qa_pairs):
        if i % 50 == 0:
            logger.info(f"  [{config_name}] {i}/{len(qa_pairs)}")

        question = qa['question']
        expected = qa['expected_answer']

        try:
            t0 = time.time()
            chunks = retriever.search(question, top_k=top_k)
            elapsed = time.time() - t0

            overlap = keyword_overlap(expected, chunks)
            g = grade(overlap)
            grades[g] += 1
            total_overlap += overlap

            records.append({
                'id': qa.get('id'),
                'overlap': round(overlap, 4),
                'grade': g,
                'latency_ms': round(elapsed * 1000),
                'chunks_returned': len(chunks),
            })

        except Exception as e:
            logger.warning(f"  Error on Q#{i}: {e}")
            errors += 1
            grades['fail'] += 1
            records.append({'id': qa.get('id'), 'overlap': 0.0, 'grade': 'fail', 'error': str(e)})

    n = len(qa_pairs)
    summary = {
        'config': config_name,
        'n': n,
        'good': grades['good'],
        'partial': grades['partial'],
        'fail': grades['fail'],
        'good_pct': round(grades['good'] / n * 100, 1),
        'partial_pct': round(grades['partial'] / n * 100, 1),
        'fail_pct': round(grades['fail'] / n * 100, 1),
        'avg_overlap': round(total_overlap / n, 4),
        'errors': errors,
    }

    logger.info(f"\n  Config {config_name} sonuç:")
    logger.info(f"    Good:    {summary['good_pct']}%  ({summary['good']}/{n})")
    logger.info(f"    Partial: {summary['partial_pct']}%  ({summary['partial']}/{n})")
    logger.info(f"    Fail:    {summary['fail_pct']}%  ({summary['fail']}/{n})")
    logger.info(f"    Avg Overlap: {summary['avg_overlap']}")

    return {'summary': summary, 'records': records}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Kaç soru test edilsin (default: tümü)')
    parser.add_argument('--top-k', type=int, default=5, help='Retrieval top-k (default: 5)')
    parser.add_argument('--configs', type=str, default='A,B,C,D', help='Çalıştırılacak configler (default: A,B,C,D)')
    parser.add_argument('--bm25-weight', type=float, default=0.5, help='Hybrid Config D BM25 weight (default: 0.5)')
    parser.add_argument('--output', type=str, default='/app/data/retrieval_comparison.json')
    args = parser.parse_args()

    run_configs = set(args.configs.upper().split(','))

    # ── Q&A dataset ──────────────────────────────────────────────────────────
    qa_path = Path(QA_DATASET_PATH)
    if not qa_path.exists():
        logger.error(f"QA dataset bulunamadı: {qa_path}")
        sys.exit(1)

    with open(qa_path, encoding='utf-8') as f:
        qa_pairs = json.load(f)

    if args.limit:
        qa_pairs = qa_pairs[:args.limit]

    logger.info(f"Q&A çiftleri yüklendi: {len(qa_pairs)}")

    # ── Qdrant & shared components ────────────────────────────────────────────
    qdrant = QdrantDBClient(host=QDRANT_HOST, port=QDRANT_PORT, collection_name=COLLECTION)
    embedder = EmbeddingsGenerator(device='cpu')

    logger.info("Qdrant'tan chunk'lar çekiliyor...")
    all_docs = []
    offset = None
    while True:
        results, next_offset = qdrant.client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
        )
        for point in results:
            payload = point.payload or {}
            all_docs.append({
                'id': str(point.id),
                'content': payload.get('text', ''),
                'metadata': payload,
            })
        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"  Toplam chunk: {len(all_docs)}")

    # ── Retriever init ────────────────────────────────────────────────────────
    results = {}

    if 'A' in run_configs:
        retriever_a = SemanticRetriever(qdrant, embedder)
        results['A'] = evaluate_config(retriever_a, qa_pairs, 'A (Semantic-only)', args.top_k)

    if 'B' in run_configs:
        bm25 = BM25Index()
        bm25.add_documents(all_docs)
        doc_meta = {d['id']: d['metadata'] for d in all_docs}
        retriever_b = BM25Retriever(bm25, doc_meta)
        results['B'] = evaluate_config(retriever_b, qa_pairs, 'B (BM25-only)', args.top_k)

    if 'C' in run_configs:
        retriever_c = TFIDFRetriever(all_docs)
        results['C'] = evaluate_config(retriever_c, qa_pairs, 'C (TF-IDF)', args.top_k)

    if 'D' in run_configs:
        sem_w = round(1.0 - args.bm25_weight, 2)
        logger.info(f"\nHybridSearcher başlatılıyor (BM25={args.bm25_weight}, Semantic={sem_w})...")
        hybrid = HybridSearcher(semantic_weight=sem_w, bm25_weight=args.bm25_weight)
        retriever_d = HybridRetriever(hybrid)
        label = f'D (Hybrid BM25={args.bm25_weight}/Sem={sem_w})'
        results['D'] = evaluate_config(retriever_d, qa_pairs, label, args.top_k)

    # ── Comparison table ──────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("RETRIEVAL KONFİGÜRASYON KARŞILAŞTIRMASI")
    logger.info("="*70)
    logger.info(f"{'Config':<25} {'Good%':>7} {'Partial%':>9} {'Fail%':>7} {'AvgOverlap':>12}")
    logger.info("-"*70)
    for k in ['A', 'B', 'C', 'D']:
        if k not in results:
            continue
        s = results[k]['summary']
        logger.info(
            f"  {s['config']:<23} {s['good_pct']:>7} {s['partial_pct']:>9} {s['fail_pct']:>7} {s['avg_overlap']:>12}"
        )
    logger.info("="*70)

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        'meta': {
            'n_questions': len(qa_pairs),
            'top_k': args.top_k,
            'collection': COLLECTION,
            'total_chunks': len(all_docs),
        },
        'configs': {k: v['summary'] for k, v in results.items()},
        'records': {k: v['records'] for k, v in results.items()},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSonuçlar kaydedildi: {out_path}")


if __name__ == '__main__':
    main()
