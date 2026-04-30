#!/usr/bin/env python3
"""
Build Q&A evaluation dataset from Freshdesk tickets.

Filters tickets for technical Q&A pairs suitable for RAG evaluation.
Output: JSON + CSV files ready for RAGAS or manual evaluation.

Usage:
    python scripts/build_qa_dataset.py
    python scripts/build_qa_dataset.py --min-solution-len 150 --lang en
    python scripts/build_qa_dataset.py --output /app/data/qa_dataset.json
"""
import os
import re
import sys
import json
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database.mongo_client import MongoDBClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Solution texts that indicate no real technical answer
LOGISTIC_PATTERNS = [
    r'send.*back', r'return.*tool', r'ship.*to', r'send.*unit',
    r'please.*return', r'envoy', r'retour', r'zurück.*schicken',
    r'renvoy', r'send.*service center', r'bring.*to.*lab',
    r'given.*to.*leo', r'donné.*au.*leo', r'given.*for.*analysis',
    r'sent.*for.*repair', r'collect.*tool',
    r'\bwarranty\b', r'under warranty', r'warranty claim',
]

INFO_REQUEST_PATTERNS = [
    r'need.*more.*information', r'additional.*information.*request',
    r'can you.*provide', r'please.*provide.*more',
    r'more.*details.*needed', r'need.*serial.*number',
    r'what.*firmware.*version', r'please.*confirm',
    r'avez.vous', r'pouvez.vous.*envoyer',
    r'check your ticket status',
]

# Products/systems the RAG is not ready for — exclude these tickets
OUT_OF_SCOPE_PATTERNS = [
    r'\brapid\b', r'\bnexonar\b', r'\bpivotware\b',
    r'\bscrew.?feed', r'\bscrewfeeder\b', r'\bfeeder\b',
]


TECHNICAL_INDICATORS = [
    r'firmware', r'update', r's-center', r'scenter',
    r'e\d{3,4}', r'w\d{3,4}', r'error code',
    r'calibrat', r'configur', r'reset', r'restart',
    r'replace', r'replac', r'remplac',
    r'check.*setting', r'verif', r'inspect',
    r'step \d', r'\d+\.\s+\w', r'follow.*procedure',
    r'download', r'install', r'connect.*unit',
    r'edock', r'eDOCK', r'screw', r'torque',
    r'battery.*charg', r'pairing', r'wifi.*setting',
]


def is_out_of_scope(ticket: dict) -> bool:
    """Returns True if ticket is about a product/system the RAG is not ready for."""
    text = (
        ticket.get('title', '') + ' ' +
        (ticket.get('description') or {}).get('content', '') if isinstance(ticket.get('description'), dict)
        else ticket.get('title', '') + ' ' + (ticket.get('description') or '')
    ).lower()
    return any(re.search(p, text) for p in OUT_OF_SCOPE_PATTERNS)


def is_logistic_reply(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in LOGISTIC_PATTERNS)


def is_info_request(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INFO_REQUEST_PATTERNS)


def has_technical_content(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in TECHNICAL_INDICATORS)


def build_question(ticket: dict) -> str:
    title = ticket.get('title', '').strip()
    desc = ''
    description = ticket.get('description')
    if description and isinstance(description, dict):
        desc = description.get('content', '').strip()
    elif isinstance(description, str):
        desc = description.strip()

    if desc and len(desc) > 20 and desc.lower() != title.lower():
        return f"{title}\n\n{desc[:500]}"
    return title


def filter_tickets(tickets: list, min_solution_len: int, lang: str) -> list:
    accepted = []
    stats = {
        'total': len(tickets),
        'out_of_scope': 0,
        'no_solution': 0,
        'short_solution': 0,
        'logistic': 0,
        'info_request': 0,
        'no_technical': 0,
        'wrong_lang': 0,
        'not_resolved': 0,
        'accepted': 0,
    }

    for t in tickets:
        solution = t.get('solution') or ''
        ticket_type = t.get('ticket_type', '')
        status = t.get('status', '')
        language = t.get('language', 'en')

        # Must be technical problem
        if ticket_type != 'technical_problem':
            continue

        # Skip out-of-scope products (RAPID, Nexonar, Pivotware, Screwfeeding)
        if is_out_of_scope(t):
            stats['out_of_scope'] += 1
            continue

        # Language filter
        if lang and language != lang:
            stats['wrong_lang'] += 1
            continue

        # Must have solution
        if not solution:
            stats['no_solution'] += 1
            continue

        # Solution length
        if len(solution) < min_solution_len:
            stats['short_solution'] += 1
            continue

        # Skip logistic replies
        if is_logistic_reply(solution):
            stats['logistic'] += 1
            continue

        # Skip info requests
        if is_info_request(solution):
            stats['info_request'] += 1
            continue

        # Must have technical content
        if not has_technical_content(solution):
            stats['no_technical'] += 1
            continue

        accepted.append(t)
        stats['accepted'] += 1

    return accepted, stats


def build_dataset(tickets: list) -> list:
    dataset = []
    for t in tickets:
        solution = t.get('solution', '')
        # Clean up solution — remove "Check your ticket status" footer
        solution = re.sub(r'Check your ticket status.*$', '', solution, flags=re.DOTALL | re.IGNORECASE).strip()
        solution = re.sub(r'BR/\w+.*$', '', solution, flags=re.DOTALL).strip()

        qa = {
            'id': t.get('ticket_id'),
            'question': build_question(t),
            'expected_answer': solution,
            'language': t.get('language', 'en'),
            'ticket_type': t.get('ticket_type'),
            'status': t.get('status'),
            'related_models': t.get('related_models', []),
            'tags': t.get('tags', []),
            'source_url': t.get('url', ''),
            'created_at': t.get('created_at'),
        }
        dataset.append(qa)

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Build Q&A evaluation dataset from tickets')
    parser.add_argument('--min-solution-len', type=int, default=120,
                        help='Minimum solution length in characters (default: 120)')
    parser.add_argument('--lang', type=str, default='en',
                        help='Language filter: en/fr/de/tr/all (default: en)')
    parser.add_argument('--output', type=str, default='/app/data/qa_dataset.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    lang = None if args.lang == 'all' else args.lang

    # Connect to MongoDB
    db = MongoDBClient()
    db.connect()
    tickets_col = db.get_tickets_collection()
    tickets = list(tickets_col.find({}, {'_id': 0}))
    logger.info(f"Loaded {len(tickets)} tickets from MongoDB")

    # Filter
    accepted, stats = filter_tickets(tickets, args.min_solution_len, lang)
    logger.info(f"\nFilter results:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Build dataset
    dataset = build_dataset(accepted)

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(dataset)} Q&A pairs to {output_path}")

    # Save CSV
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'question', 'expected_answer', 'language', 'status', 'related_models', 'source_url'])
        writer.writeheader()
        for qa in dataset:
            writer.writerow({
                'id': qa['id'],
                'question': qa['question'],
                'expected_answer': qa['expected_answer'][:300],
                'language': qa['language'],
                'status': qa['status'],
                'related_models': ', '.join(qa['related_models']),
                'source_url': qa['source_url'],
            })
    logger.info(f"Saved CSV to {csv_path}")

    # Preview
    if dataset:
        logger.info(f"\n--- Sample Q&A ---")
        sample = dataset[0]
        logger.info(f"Q: {sample['question'][:120]}")
        logger.info(f"A: {sample['expected_answer'][:120]}")


if __name__ == '__main__':
    main()
