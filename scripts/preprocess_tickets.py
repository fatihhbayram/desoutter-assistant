import json
import logging
import sys
from pathlib import Path

# Add paths for Docker container
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/src")
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.scraper.ticket_preprocessor import TicketPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load raw tickets from MongoDB export (ALL tickets)
    raw_path = Path("data/tickets/all_tickets_raw.json")
    output_path = Path("data/tickets/all_tickets_cleaned.json")
    
    logger.info(f"Loading tickets from {raw_path}")
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_tickets = json.load(f)
    logger.info(f"Loaded {len(raw_tickets)} raw tickets")
    
    # Process all loaded tickets (already limited to 200)
    preprocessor = TicketPreprocessor(
        min_question_length=20,
        min_answer_length=50
    )
    cleaned, stats = preprocessor.process_batch(raw_tickets)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved {len(cleaned)} cleaned tickets to {output_path}")
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Input:           {stats.total_input}")
    print(f"Output:          {stats.passed} ({stats.passed/stats.total_input*100:.1f}%)")
    print(f"Filtered:")
    print(f"  - Bulletin actions: {stats.filtered_bulletin}")
    print(f"  - Feedback only:    {stats.filtered_feedback}")
    print(f"  - Duplicates:       {stats.filtered_duplicate}")
    print(f"  - Low quality:      {stats.filtered_low_quality}")
    print("="*60)

if __name__ == "__main__":
    main()
