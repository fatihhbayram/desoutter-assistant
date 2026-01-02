#!/usr/bin/env python3
"""
=============================================================================
Low-Quality Chunk Cleanup Script
=============================================================================
Removes low-quality chunks from ChromaDB that negatively affect retrieval.

Quality checks:
1. Too short (< 50 chars)
2. Too few words (< 5 words)
3. No alphabetic characters
4. Too many newlines (formatting noise)
5. Repetitive content
6. Empty or whitespace-only

Usage:
    # Dry run (preview only)
    python scripts/clean_low_quality_chunks.py --dry-run
    
    # Actually delete
    python scripts/clean_low_quality_chunks.py
    
    # With verbose output
    python scripts/clean_low_quality_chunks.py --verbose
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectordb.chroma_client import ChromaDBClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChunkQualityChecker:
    """Check chunk quality and identify low-quality content"""
    
    # Minimum thresholds
    MIN_CHARS = 50
    MIN_WORDS = 5
    MAX_NEWLINE_RATIO = 0.15  # Max 15% newlines
    MAX_DIGIT_RATIO = 0.7    # Max 70% digits (likely tables/numbers only)
    
    def __init__(self):
        self.stats = {
            'total': 0,
            'too_short': 0,
            'few_words': 0,
            'no_letters': 0,
            'too_many_newlines': 0,
            'mostly_digits': 0,
            'repetitive': 0,
            'kept': 0
        }
    
    def is_low_quality(self, content: str) -> tuple[bool, str]:
        """
        Check if chunk is low quality.
        
        Args:
            content: Chunk text content
            
        Returns:
            (is_low_quality: bool, reason: str)
        """
        self.stats['total'] += 1
        
        if not content or not content.strip():
            return True, 'empty'
        
        content = content.strip()
        
        # 1. Too short
        if len(content) < self.MIN_CHARS:
            self.stats['too_short'] += 1
            return True, f'too_short ({len(content)} chars)'
        
        # 2. Too few words
        words = content.split()
        if len(words) < self.MIN_WORDS:
            self.stats['few_words'] += 1
            return True, f'few_words ({len(words)} words)'
        
        # 3. No alphabetic characters
        alpha_count = sum(1 for c in content if c.isalpha())
        if alpha_count == 0:
            self.stats['no_letters'] += 1
            return True, 'no_letters'
        
        # 4. Too many newlines (formatting noise)
        newline_ratio = content.count('\n') / len(content) if content else 0
        if newline_ratio > self.MAX_NEWLINE_RATIO:
            self.stats['too_many_newlines'] += 1
            return True, f'too_many_newlines ({newline_ratio:.2%})'
        
        # 5. Mostly digits (likely just numbers/tables)
        digit_count = sum(1 for c in content if c.isdigit())
        total_alnum = alpha_count + digit_count
        if total_alnum > 0:
            digit_ratio = digit_count / total_alnum
            if digit_ratio > self.MAX_DIGIT_RATIO:
                self.stats['mostly_digits'] += 1
                return True, f'mostly_digits ({digit_ratio:.2%})'
        
        # 6. Repetitive content (same word repeated many times)
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        if len(word_counts) > 0:
            max_repeat = max(word_counts.values())
            repeat_ratio = max_repeat / len(words)
            if repeat_ratio > 0.5 and len(words) > 10:  # Same word >50% of text
                self.stats['repetitive'] += 1
                return True, f'repetitive ({repeat_ratio:.2%})'
        
        # Passed all checks
        self.stats['kept'] += 1
        return False, 'ok'
    
    def get_stats_summary(self) -> str:
        """Get summary of quality check statistics"""
        total = self.stats['total']
        if total == 0:
            return "No chunks processed"
        
        removed = total - self.stats['kept']
        
        lines = [
            f"\n{'='*60}",
            f"CHUNK QUALITY ANALYSIS",
            f"{'='*60}",
            f"Total chunks analyzed: {total}",
            f"Chunks to keep: {self.stats['kept']} ({self.stats['kept']/total*100:.1f}%)",
            f"Chunks to remove: {removed} ({removed/total*100:.1f}%)",
            f"\nRemoval breakdown:",
            f"  - Too short (<{self.MIN_CHARS} chars): {self.stats['too_short']}",
            f"  - Few words (<{self.MIN_WORDS} words): {self.stats['few_words']}",
            f"  - No letters: {self.stats['no_letters']}",
            f"  - Too many newlines: {self.stats['too_many_newlines']}",
            f"  - Mostly digits: {self.stats['mostly_digits']}",
            f"  - Repetitive: {self.stats['repetitive']}",
            f"{'='*60}"
        ]
        return '\n'.join(lines)


def clean_chunks(dry_run: bool = True, verbose: bool = False):
    """
    Clean low-quality chunks from ChromaDB.
    
    Args:
        dry_run: If True, only preview without deleting
        verbose: If True, show each deleted chunk
    """
    print(f"\n{'='*60}")
    print("CHROMADB CHUNK CLEANUP")
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else '⚠️  ACTUAL DELETE'}")
    print(f"{'='*60}\n")
    
    # Initialize
    client = ChromaDBClient()
    checker = ChunkQualityChecker()
    
    collection = client.collection
    total_count = collection.count()
    
    print(f"Total chunks in ChromaDB: {total_count}")
    
    if total_count == 0:
        print("No chunks to process")
        return
    
    # Fetch all chunks in batches
    batch_size = 1000
    low_quality_ids = []
    
    for offset in range(0, total_count, batch_size):
        print(f"Processing batch {offset//batch_size + 1}...")
        
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )
        
        for i, doc_id in enumerate(results['ids']):
            content = results['documents'][i] if results['documents'] else ''
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            
            is_low, reason = checker.is_low_quality(content)
            
            if is_low:
                low_quality_ids.append(doc_id)
                
                if verbose:
                    source = metadata.get('source', 'unknown')[:40]
                    preview = content[:60].replace('\n', ' ') if content else ''
                    print(f"  ❌ [{reason}] {source}: {preview}...")
    
    # Print summary
    print(checker.get_stats_summary())
    
    if not low_quality_ids:
        print("\n✅ No low-quality chunks found!")
        return
    
    # Delete if not dry run
    if dry_run:
        print(f"\n🔍 DRY RUN: Would delete {len(low_quality_ids)} chunks")
        print("Run without --dry-run to actually delete")
    else:
        print(f"\n⚠️  Deleting {len(low_quality_ids)} low-quality chunks...")
        
        # Delete in batches
        batch_size = 100
        for i in range(0, len(low_quality_ids), batch_size):
            batch_ids = low_quality_ids[i:i+batch_size]
            collection.delete(ids=batch_ids)
            print(f"  Deleted batch {i//batch_size + 1} ({len(batch_ids)} chunks)")
        
        # Verify
        new_count = collection.count()
        print(f"\n✅ Cleanup complete!")
        print(f"  Before: {total_count} chunks")
        print(f"  After: {new_count} chunks")
        print(f"  Removed: {total_count - new_count} chunks")


def main():
    parser = argparse.ArgumentParser(description='Clean low-quality chunks from ChromaDB')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Preview without deleting (default: True)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete chunks (disables dry-run)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show each chunk being removed'
    )
    
    args = parser.parse_args()
    
    # If --execute specified, disable dry_run
    dry_run = not args.execute
    
    clean_chunks(dry_run=dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
